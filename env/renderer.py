import numpy as np
import pygame
import math
import random
from env.config import Config
"""
This script defines the `Renderer` class, which is responsible for all visual
output of the environment using the Pygame library.

The class handles:
- Lazy initialization of the Pygame window.
- Drawing the static grid and background.
- Rendering all dynamic game objects, including UAVs, fires, obstacles, and smoke areas.
- Visualizing dynamic information like sensor ranges, smoke impairment zones, and health bars.
- Rendering temporary visual effects such as dousing a fire with water.
- Displaying a Heads-Up Display (HUD) with key simulation metrics.
"""

class Renderer:
    """
    Manages the rendering of the UAVs environment using Pygame.

    This class encapsulates all drawing logic, from the background grid to the
    individual game objects and visual effects. It is instantiated by `FireFightingEnv`
    when `render_mode` is set to 'human'.
    """
    def __init__(self, config):
        """
        Initializes the Renderer.

        This does not create the Pygame window immediately. The window is created
        on the first call to `render_frame` (lazy initialization).

        Args:
            config (Config): The configuration object containing rendering parameters
                             like colors, sizes, and dimensions.
        """
        pygame.init()
        pygame.font.init()
        self.config = config
        self.screen = None
        self.clock = None
        self.font = pygame.font.SysFont("monospace", 16)

        self.width = config.GRID_SIZE * config.CELL_SIZE
        self.height = config.GRID_SIZE * config.CELL_SIZE + 50

        self.is_recording = False
        self.recorded_frames = []

    def _initialize_pygame_for_human_mode(self):
        """
        Initializes Pygame, the display window, and the font if they haven't been already.

        This method is called internally by `render_frame` to ensure that Pygame is
        only initialized when rendering is actually required.
        """
        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AFRL RIS 2026 Challenge Competition")
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render_frame(self, env):
        """
        Renders a single frame of the simulation.

        This is the main public method of the class. It orchestrates the drawing of
        all environment components in the correct order, from background to foreground,
        and then updates the display.

        Args:
            env (FireFightingEnv): The environment instance containing the state to be rendered.
        """

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(self.config.COLOR_BACKGROUND)

        self._draw_grid(canvas)
        for obstacle in env.obstacles:
            self._draw_obstacle(canvas, obstacle)
        self._draw_smoke_ranges(canvas, env.dense_smoke_areas)
        for smoke in env.dense_smoke_areas:
            self._draw_smoke(canvas, smoke)
        for fire in env.fires:
            if not fire.is_extinguished:
                self._draw_fire(canvas, fire)
        self._draw_uav_sensors(canvas, env.uavs, env.step_count)
        for uav in env.uavs:
            self._draw_uav(canvas, uav)
        for effect in env.effects:
            self._draw_effect(canvas, effect)
        self._draw_hud(canvas, env)

        return canvas

    
    def render_human(self, env):
        """Render for human consumption: draw to screen and handle recording."""
        if not pygame.get_init() or not self.screen:
            self._initialize_pygame_for_human_mode()
        
        canvas = self.render_frame(env)
        
        # Blit the final canvas to the visible screen
        self.screen.blit(canvas, (0, 0))

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(env.metadata["render_fps"])

    def render_rgb_array(self, env):
        """Render for programmatic consumption: return a NumPy array."""
        canvas = self.render_frame(env)
        
        frame = pygame.surfarray.array3d(canvas)
        return frame.transpose(1, 0, 2) # Return in standard (H, W, C) format

    def _draw_grid(self, canvas):
        """Draws the background grid lines on the canvas."""
        for x in range(self.config.GRID_SIZE + 1):
            pygame.draw.line(canvas, self.config.COLOR_GRID, (x * self.config.CELL_SIZE, 0),
                             (x * self.config.CELL_SIZE, self.height - 50))
        for y in range(self.config.GRID_SIZE + 1):
            pygame.draw.line(canvas, self.config.COLOR_GRID, (0, y * self.config.CELL_SIZE),
                             (self.width, y * self.config.CELL_SIZE))

    def _get_center_px(self, pos):
        """
        Converts grid coordinates to pixel coordinates for the center of a cell.

        Args:
            pos (tuple or np.ndarray): The (x, y) grid coordinates.

        Returns:
            tuple: The (x, y) pixel coordinates.
        """
        return pos[0] * self.config.CELL_SIZE + self.config.CELL_SIZE / 2, pos[
            1] * self.config.CELL_SIZE + self.config.CELL_SIZE / 2

    def _draw_obstacle(self, canvas, obstacle):
        """Draws a static, noticeable rocky outcrop obstacle."""
        if not hasattr(obstacle, 'shape_points'):
            center_x, center_y = self._get_center_px(obstacle.pos)

            # This random generation now runs ONLY ONCE per obstacle's lifetime.
            points = []
            num_points = 7  # More points = more jagged
            for i in range(num_points):
                # Using a pseudo-random but deterministic seed could also work,
                # but storing the points is the most robust method.
                # We'll use a simple random generation for unique shapes.
                import random
                angle = (360 / num_points) * i + random.uniform(-15, 15)
                radius = (self.config.CELL_SIZE / 2) * random.uniform(0.8, 1.2)

                x = center_x + radius * math.cos(math.radians(angle))
                y = center_y + radius * math.sin(math.radians(angle))
                points.append((x, y))

            # Store the generated points ON the obstacle object itself.
            obstacle.shape_points = points

        # --- Drawing Logic (uses the stored points) ---
        # This part runs every frame, but always uses the smokee `shape_points`.

        # Shadow layer (drawn first, slightly offset)
        shadow_points = [(p[0] + 3, p[1] + 3) for p in obstacle.shape_points]
        pygame.draw.polygon(canvas, self.config.COLOR_ROCK_SHADOW, shadow_points)

        # Main rock face
        pygame.draw.polygon(canvas, self.config.COLOR_ROCK, obstacle.shape_points)

        # Highlight layer (a smaller polygon on top)
        # We still need the center, so we can calculate it once.
        center_x, center_y = self._get_center_px(obstacle.pos)
        highlight_points = [(center_x + (p[0] - center_x) * 0.7, center_y + (p[1] - center_y) * 0.7) for p in
                            obstacle.shape_points]
        pygame.draw.polygon(canvas, self.config.COLOR_ROCK_HIGHLIGHT, highlight_points)

    def _draw_smoke(self, canvas, smoke):
        """Draws the smoke area as a dense, smokey core."""
        center_x, center_y = self._get_center_px(smoke.pos)

        # --- Smokey Core Generation ---
        # We will draw a high concentration of particles in a small area.

        # A smaller radius for the core smoke effect
        core_radius = self.config.CELL_SIZE / 1.5

        # More particles in a smaller area = denser smoke
        num_core_particles = 75

        # It's best to use a slightly darker/more opaque color for the core
        # to make it stand out from the wider, more transparent range smoke.
        # e.g., COLOR_SMOKE_CORE = (50, 50, 50, 50)
        core_color = self.config.COLOR_SMOKE_CORE

        for _ in range(num_core_particles):
            # Generate a particle within the smaller 'core_radius'
            offset_x = random.uniform(-core_radius, core_radius)
            offset_y = random.uniform(-core_radius, core_radius)

            # Ensure the particle is within a circular area (optional but looks better)
            if offset_x ** 2 + offset_y ** 2 < core_radius ** 2:
                pos_x = center_x + offset_x
                pos_y = center_y + offset_y

                particle_radius = random.randint(5, 15)

                # We draw directly onto the main canvas here, as _draw_smoke_ranges
                # will draw the larger, more transparent smoke underneath it.
                # Using BLEND_RGBA_ADD is great if your Pygame version supports it.
                # If not, remove the 'blend' argument.
                try:
                    pygame.draw.circle(canvas, core_color, (pos_x, pos_y), particle_radius, blend=pygame.BLEND_RGBA_ADD)
                except TypeError:
                    pygame.draw.circle(canvas, core_color, (pos_x, pos_y), particle_radius)

    def _draw_smoke_ranges(self, canvas, smokes):
        """Draws the semi-transparent impairment ranges for all smoke areas. """
        smoke_range_surface = pygame.Surface((self.width, self.height - 50), pygame.SRCALPHA)

        for smoke in smokes:
            center_x, center_y = self._get_center_px(smoke.pos)

            range_px = smoke.radius * self.config.CELL_SIZE
            rect_left = center_x - range_px
            rect_top = center_y - range_px
            rect_width = range_px * 2

            num_particles = 200

            for _ in range(num_particles):
                pos_x = random.uniform(rect_left, rect_left + rect_width)
                pos_y = random.uniform(rect_top, rect_top + rect_width)

                particle_radius = random.randint(10, 25)
                color = self.config.COLOR_SMOKE_RANGE

                # THE FIX: The 'blend' keyword argument has been removed from this line
                pygame.draw.circle(smoke_range_surface, color, (pos_x, pos_y), particle_radius)

        # Blit the entire smoke surface onto the main canvas
        canvas.blit(smoke_range_surface, (0, 0))

    def _draw_fire(self, canvas, fire):

        """Draws a single fire as a fire effect with a health bar above it."""
        center_x, center_y = self._get_center_px(fire.pos)

        # Health bar (remains unchanged)
        health_pct = max(0, fire.hp / fire.max_hp)
        pygame.draw.rect(canvas, self.config.COLOR_HEALTH_BAR_BG, (
            center_x - self.config.CELL_SIZE / 2, center_y - self.config.CELL_SIZE / 2 - 8, self.config.CELL_SIZE, 5))
        if health_pct > 0:
            pygame.draw.rect(canvas, self.config.COLOR_HEALTH_BAR_FG, (
                center_x - self.config.CELL_SIZE / 2, center_y - self.config.CELL_SIZE / 2 - 8,
                self.config.CELL_SIZE * health_pct, 5))

        # --- Fire Shape (replaces the Star shape) ---
        flame_height = self.config.CELL_SIZE * 1.2

        if fire.known:
            # Draw a full, bright, animated fire for a "found" fire
            flicker_x = random.uniform(-self.config.CELL_SIZE * 0.1, self.config.CELL_SIZE * 0.1)
            flicker_height = random.uniform(0, self.config.CELL_SIZE * 0.15)

            # 1. Outer Flame (Red)
            outer_flame = [
                (center_x - self.config.CELL_SIZE / 2.2, center_y + self.config.CELL_SIZE / 2),
                (center_x + self.config.CELL_SIZE / 2.2, center_y + self.config.CELL_SIZE / 2),
                (center_x + flicker_x, center_y - flame_height / 2 - flicker_height)
            ]
            pygame.draw.polygon(canvas, self.config.COLOR_FIRE_RED, outer_flame)

            # 2. Middle Flame (Orange)
            middle_flame = [
                (center_x - self.config.CELL_SIZE / 4, center_y + self.config.CELL_SIZE / 2),
                (center_x + self.config.CELL_SIZE / 4, center_y + self.config.CELL_SIZE / 2),
                (center_x + flicker_x * 0.8, center_y - flame_height / 2.2 - flicker_height)
            ]
            pygame.draw.polygon(canvas, self.config.COLOR_FIRE_ORANGE, middle_flame)

            # 3. Inner Flame (Yellow)
            inner_flame = [
                (center_x - self.config.CELL_SIZE / 8, center_y + self.config.CELL_SIZE / 2),
                (center_x + self.config.CELL_SIZE / 8, center_y + self.config.CELL_SIZE / 2),
                (center_x + flicker_x * 0.6, center_y - flame_height / 3 - flicker_height)
            ]
            pygame.draw.polygon(canvas, self.config.COLOR_FIRE_YELLOW, inner_flame)
        else:
            # Draw a simple, static, dark flame for a "hidden" fire
            hidden_flame_points = [
                (center_x - self.config.CELL_SIZE / 2.5, center_y + self.config.CELL_SIZE / 2),
                (center_x + self.config.CELL_SIZE / 2.5, center_y + self.config.CELL_SIZE / 2),
                (center_x, center_y - flame_height / 2)
            ]
            pygame.draw.polygon(canvas, self.config.COLOR_FIRE_HIDDEN, hidden_flame_points)

    def _draw_uav(self, canvas, uav):
        """Draws a single UAV/drone with spinning propellers."""
        center_x, center_y = self._get_center_px(uav.pos)
        size = self.config.CELL_SIZE
        fuselage = pygame.Rect(-size * 0.15, -size * 0.3, size * 0.3, size * 0.6)

        arm_length = size / 2
        prop_positions = [
            (-arm_length, -arm_length),
            (arm_length, -arm_length),
            (-arm_length, arm_length),
            (arm_length, arm_length)
        ]

        prop_angle = (pygame.time.get_ticks() / 5) % 360  # Fast spin
        rot_angle_map = {0: 0, 1: 270, 2: 180, 3: 90}
        rotation_rad = math.radians(rot_angle_map[uav.orientation])
        cos_rot, sin_rot = math.cos(rotation_rad), math.sin(rotation_rad)

        def rotate_and_translate(p):
            x, y = p
            new_x = x * cos_rot - y * sin_rot
            new_y = x * sin_rot + y * cos_rot
            return (new_x + center_x, new_y + center_y)

        prop_blur_color = (*self.config.COLOR_UAV, 100)
        prop_line_color = (min(self.config.COLOR_UAV[0] + 50, 255),
                           min(self.config.COLOR_UAV[1] + 50, 255),
                           min(self.config.COLOR_UAV[2] + 50, 255))

        for pos in prop_positions:
            p_center = rotate_and_translate(pos)

            blade_radius = size / 4
            start_offset = (
            blade_radius * math.cos(math.radians(prop_angle)), blade_radius * math.sin(math.radians(prop_angle)))
            end_offset = (-start_offset[0], -start_offset[1])
            pygame.draw.line(canvas, prop_line_color, (p_center[0] + start_offset[0], p_center[1] + start_offset[1]),
                             (p_center[0] + end_offset[0], p_center[1] + end_offset[1]), 2)
            pygame.draw.circle(canvas, prop_blur_color, p_center, blade_radius, 1)

        for pos in prop_positions:
            arm_end = rotate_and_translate(pos)
            pygame.draw.line(canvas, self.config.COLOR_UAV, (center_x, center_y), arm_end, 3)

        fuselage_points = [
            rotate_and_translate(fuselage.topleft),
            rotate_and_translate(fuselage.topright),
            rotate_and_translate(fuselage.bottomright),
            rotate_and_translate(fuselage.bottomleft)
        ]
        pygame.draw.polygon(canvas, self.config.COLOR_UAV, fuselage_points)

        front_indicator_pos = rotate_and_translate((0, -size * 0.2))
        pygame.draw.circle(canvas, prop_line_color, front_indicator_pos, 3)

    def _draw_uav_sensors(self, canvas, uavs, step_count):
        """Draws the pulsating sensor range for all uavs."""
        sensor_surface = pygame.Surface((self.width, self.height - 50), pygame.SRCALPHA)
        pulse_alpha = 30 + 20 * math.sin(step_count * 0.2)
        for uav in uavs:
            pygame.draw.circle(sensor_surface, (*self.config.COLOR_VISION_PULSE[:3], pulse_alpha),
                               self._get_center_px(uav.pos), self.config.UAV_VISION_RANGE * self.config.CELL_SIZE)
        canvas.blit(sensor_surface, (0, 0))

    def _draw_effect(self, canvas, effect):
        """
        Draws temporary visual effects like dousing a fire with water, or smoke.

        Args:
            canvas (pygame.Surface): The surface to draw on.
            effect (dict): A dictionary describing the effect to draw.
        """
        # print(effect)

        if effect['type'] == 'water_splash':
            # Main radius of the splash
            base_radius = self.config.CELL_SIZE * (effect['timer'] / 5)

            # Create a surface for transparency effects
            splash_surface = pygame.Surface((self.config.CELL_SIZE * 2, self.config.CELL_SIZE * 2), pygame.SRCALPHA)

            # Draw multiple circles to simulate ripples
            for i in range(3):
                # Ripples are spaced out
                radius = base_radius - (i * (self.config.CELL_SIZE / 5))
                if radius > 0:
                    # Alpha (transparency) decreases for outer ripples and as time passes
                    alpha = 200 - (effect['timer'] * 40) - (i * 40)
                    if alpha > 0:
                        color = (*self.config.COLOR_WATER, alpha)  # Add alpha to the color
                        # Draw the ripple. Note that we draw on the new surface at its center.
                        pygame.draw.circle(splash_surface, color, (self.config.CELL_SIZE, self.config.CELL_SIZE),
                                           int(radius), width=2)

            pos_px = self._get_center_px(effect['pos'])
            canvas.blit(splash_surface, (pos_px[0] - self.config.CELL_SIZE, pos_px[1] - self.config.CELL_SIZE))

        elif effect['type'] == 'smoke_impairment':
            end_px = self._get_center_px(effect['end'])
            pygame.draw.circle(canvas, self.config.COLOR_FIRE_RED, end_px, radius=5)

    def _draw_legend_uav(self, canvas, center_x, center_y, size):
        """Draws a static UAV icon for the legend."""
        fuselage_w, fuselage_h = size * 0.3, size * 0.6
        fuselage = pygame.Rect(center_x - fuselage_w / 2, center_y - fuselage_h / 2, fuselage_w, fuselage_h)
        pygame.draw.rect(canvas, self.config.COLOR_UAV, fuselage, border_radius=3)

        arm_length = size / 2
        arm_thickness = 2
        pygame.draw.line(canvas, self.config.COLOR_UAV, (center_x - arm_length, center_y - arm_length),
                         (center_x + arm_length, center_y + arm_length), arm_thickness)
        pygame.draw.line(canvas, self.config.COLOR_UAV, (center_x + arm_length, center_y - arm_length),
                         (center_x - arm_length, center_y + arm_length), arm_thickness)

        prop_radius = size * 0.15
        prop_positions = [
            (center_x - arm_length, center_y - arm_length), (center_x + arm_length, center_y - arm_length),
            (center_x - arm_length, center_y + arm_length), (center_x + arm_length, center_y + arm_length)
        ]
        for pos in prop_positions:
            pygame.draw.circle(canvas, self.config.COLOR_UAV, pos, prop_radius)

    def _draw_legend_fire(self, canvas, center_x, center_y, size):
        """Draws a layered fire icon for the legend."""
        flame_height = size * 1.2
        outer_points = [(center_x - size / 2, center_y + size / 2), (center_x + size / 2, center_y + size / 2),
                        (center_x, center_y - flame_height / 2)]
        pygame.draw.polygon(canvas, self.config.COLOR_FIRE_RED, outer_points)
        middle_points = [(center_x - size / 4, center_y + size / 2), (center_x + size / 4, center_y + size / 2),
                         (center_x, center_y - flame_height / 2.5)]
        pygame.draw.polygon(canvas, self.config.COLOR_FIRE_ORANGE, middle_points)
        inner_points = [(center_x - size / 8, center_y + size / 2), (center_x + size / 8, center_y + size / 2),
                        (center_x, center_y - flame_height / 4)]
        pygame.draw.polygon(canvas, self.config.COLOR_FIRE_YELLOW, inner_points)

    def _draw_legend_smoke(self, canvas, center_x, center_y, size):
        """Draws a stylized smoke cloud icon for the legend."""
        half_size = size / 2
        core_points = [
            (center_x, center_y - half_size * 0.9), (center_x + half_size * 0.8, center_y - half_size * 0.3),
            (center_x + half_size * 0.6, center_y + half_size * 0.7),
            (center_x - half_size * 0.7, center_y + half_size * 0.8),
            (center_x - half_size, center_y), (center_x - half_size * 0.3, center_y - half_size * 0.5)
        ]
        pygame.draw.polygon(canvas, self.config.COLOR_SMOKE_CORE, core_points)

        highlight_color = tuple(min(c + 40, 255) for c in self.config.COLOR_SMOKE_CORE)
        inner_points = [(center_x + (p[0] - center_x) * 0.7, center_y + (p[1] - center_y) * 0.7) for p in core_points]
        pygame.draw.polygon(canvas, highlight_color, inner_points)

    def _draw_legend_obstacle(self, canvas, center_x, center_y, size):
        """Draws a rocky obstacle icon for the legend. Generates points once and reuses."""
        if not hasattr(self, 'hud_obstacle_points'):
            points = []
            for i in range(7):
                angle = (360 / 7) * i + random.uniform(-15, 15)
                radius = (size / 2) * random.uniform(0.8, 1.2)
                x = center_x + radius * math.cos(math.radians(angle))
                y = center_y + radius * math.sin(math.radians(angle))
                points.append((x, y))
            self.hud_obstacle_points = points

        shadow_points = [(p[0] + 3, p[1] + 3) for p in self.hud_obstacle_points]
        pygame.draw.polygon(canvas, self.config.COLOR_ROCK_SHADOW, shadow_points)
        pygame.draw.polygon(canvas, self.config.COLOR_ROCK, self.hud_obstacle_points)

        highlight_points = [(center_x + (p[0] - center_x) * 0.7, center_y + (p[1] - center_y) * 0.7) for p in
                            self.hud_obstacle_points]
        pygame.draw.polygon(canvas, self.config.COLOR_ROCK_HIGHLIGHT, highlight_points)

    def _draw_hud(self, canvas, env):
        """Draws the HUD with dynamically calculated, even spacing for stats."""
        hud_height = 70
        hud_top = self.height - hud_height
        hud_area = pygame.Rect(0, hud_top, self.width, hud_height)
        pygame.draw.rect(canvas, self.config.COLOR_GRID, hud_area)

        stats_y_pos = hud_top + 5

        stats_to_render = [
            self.font.render(f"Episode: {env.current_episode}", True, self.config.COLOR_HUD_TEXT),
            self.font.render(f"Steps: {env.step_count}/{self.config.MAX_EPISODE_STEPS}", True,
                             self.config.COLOR_HUD_TEXT),
            self.font.render(f"Reward Total: {env.total_reward:.2f}", True, self.config.COLOR_HUD_TEXT),
            self.font.render(f"Fires Extinguished: {env.get_fires_extinguished()}/{len(env.fires)}", True,
                             self.config.COLOR_HUD_TEXT),
        ]

        total_text_width = sum(surf.get_width() for surf in stats_to_render)
        padding = 20
        total_empty_space = self.width - total_text_width - (2 * padding)
        num_gaps = len(stats_to_render) - 1
        if num_gaps > 0:
            gap_size = total_empty_space / num_gaps
        else:
            gap_size = 0

        current_x = padding
        for surf in stats_to_render:
            canvas.blit(surf, (current_x, stats_y_pos))
            current_x += surf.get_width() + gap_size

        legend_y = hud_top + 45
        icon_size = self.config.CELL_SIZE // 2
        icon_text_spacing = 8

        water_text_str = f"Water Remaining: {' / '.join(map(str, [uav.water_drops for uav in env.uavs]))}"
        water_surf = self.font.render(water_text_str, True, self.config.COLOR_AMMO_TEXT)
        water_y = legend_y - water_surf.get_height() / 2
        canvas.blit(water_surf, (10, water_y))

        legend_items = [
            ("UAV", self._draw_legend_uav),
            ("Fire", self._draw_legend_fire),
            ("Smoke", self._draw_legend_smoke),
            ("Obstacle", self._draw_legend_obstacle)
        ]

        est_text_width = 60
        icon_item_width = icon_size + icon_text_spacing + est_text_width
        total_icons_width = (icon_item_width * len(legend_items))
        legend_start_x = (self.width - total_icons_width) // 2

        current_x = legend_start_x
        for label, draw_func in legend_items:
            icon_center_x = current_x + icon_size / 2
            draw_func(canvas, icon_center_x, legend_y, icon_size)
            text_surf = self.font.render(label, True, self.config.COLOR_HUD_TEXT)
            text_y = legend_y - text_surf.get_height() / 2
            canvas.blit(text_surf, (current_x + icon_size + icon_text_spacing, text_y))
            current_x += icon_item_width + 20

    def close(self):
        """
        Properly quits Pygame and closes the display window.

        Should be called when the environment is closed to free up resources.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
