import numpy as np
from env.config import Config

"""
This script defines the fundamental game objects that populate the  simulation environment. 
It establishes a `GameObject` base class and extends it for specific 
entity types: `Obstacle`, `fire`, `dense smoke area`, and `UAV`.

Each class encapsulates properties and behaviors relevant to that type of entity,
such as position, health, range, and movement capabilities. These objects are
instantiated and managed by the `FireFightingEnv` class to construct the simulation state.
"""

class GameObject:
    """
    Base class for all interactive objects within the environment.

    All game objects possess a position on the grid.
    """
    def __init__(self, pos):
        """
        Initializes a new GameObject with a given position.

        Args:
            pos (tuple or list or np.ndarray): The (x, y) coordinates of the object on the grid.
        """
        self.pos = np.array(pos)


class Obstacle(GameObject):
    """
    Represents a static, impassable obstacle in the environment.

    UAVs cannot move into an obstacle's position.
    Inherits position property from GameObject.
    """
    def __init__(self, pos):
        """
        Initializes an Obstacle at a specified position.

        Args:
            pos (tuple or list or np.ndarray): The (x, y) coordinates of the obstacle.
        """
        super().__init__(pos)

class Fire(GameObject):
    """
    Represents a fire that needs to be extinguished by firefighting units.

    Fires have health points, representing their intensity, and can be marked
    as 'known' once discovered.
    """
    def __init__(self, pos, hp):
        """
        Initializes a fire with a position and an initial intensity.

        Args:
            pos (tuple or list or np.ndarray): The (x, y) coordinates of the fire.
            hp (int): The initial health points (intensity) of the fire.
        """
        super().__init__(pos)
        self.max_hp = hp
        self.hp = hp
        self.known = False

    def douse(self, amount=1):
        """
        Reduces the fire's intensity (health) by a specified amount.

        This simulates applying water or fire retardant. The intensity cannot go below zero.

        Args:
            amount (int, optional): The amount of intensity to reduce. Defaults to 1.
        """
        self.hp = max(0, self.hp - amount)

    @property
    def is_extinguished(self):
        """
        Checks if the fire has been extinguished (its intensity is zero or less).

        Returns:
            bool: True if the fire's intensity is 0 or less, False otherwise.
        """
        return self.hp <= 0


class DenseSmoke(GameObject):
    """
    Represents an area of dense smoke, a stationary environmental hazard for UAVs.

    UAVs entering this area may have their sensors or other functions impaired leading to a crash.
    """
    def __init__(self, pos, radius, impairment_factor):
        """
        Initializes a dense smoke area with its position, radius, and impairment level.

        Args:
            pos (tuple or list or np.ndarray): The (x, y) coordinates for the center of the smoke.
            radius (int or float): The radius of the smoke area.
            impairment_factor (float): The degree to which the smoke affects a UAV
                                       (e.g., 0.5 for 50% sensor reduction).
        """
        super().__init__(pos)
        self.radius = radius
        self.impairment_factor = impairment_factor

    def in_area(self, uav_pos):
        """
        Checks if a given position (e.g., a UAVs position) is within the smoke.

        Args:
            uav_pos (np.ndarray): The (x, y) coordinates of the object to check.

        Returns:
            bool: True if the uav_pos is within the smoke's radius, False otherwise.
        """
        return np.linalg.norm(self.pos - uav_pos) <= self.radius


class UAV(GameObject):
    """
    Represents Unmanned Aerial Vehicle (UAV) controlled by the agent.

    UAVss have a position, orientation, and a limited supply of water. They can
    move by changing orientation and then moving in the new direction.
    """
    def __init__(self, pos, water_drops):
        """
        Initializes a UAVs at a starting position with a given water count.

        Args:
            pos (tuple or list or np.ndarray): The (x, y) coordinates of the UAVs.
            water_count (int): The initial number of water the UAVs carries.
        """
        super().__init__(pos)
        self.orientation = 0  # 0:Up, 1:Right, 2:Down, 3:Left
        self.water_drops = water_drops

    def turn(self, direction):
        """
        Changes the UAVs orientation.

        Args:
            direction (int): -1 to turn left (counter-clockwise), 1 to turn right (clockwise).
        """
        self.orientation = (self.orientation + direction + 4) % 4

    def get_move_vector(self, straight=False):
        """
        Returns the 2D vector corresponding to the UAVs current orientation.

        This vector indicates the direction the UAVs will move if it takes a 'straight' action.

        Returns:
            np.ndarray: A 2-element numpy array representing the (dx, dy) movement.
        """
        return np.array(Config.ORIENTATION_TO_VECTOR[self.orientation])

    def has_water_drops(self):
        """
        Checks if the UAVs has any water remaining.

        Returns:
            bool: True if the UAVs has 1 or more water, False otherwise.
        """
        return self.water_drops > 0

    def expend_water(self):
        """
        Attempts to expend one water from the UAVs inventory.

        If water is available, one is deducted, and True is returned.
        If no water is available, nothing happens, and False is returned.

        Returns:
            bool: True if water was successfully expended, False otherwise.
        """
        if self.has_water_drops():
            self.water_drops -= 1
            return True
        return False

