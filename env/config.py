import numpy as np

class Config:
    """
    Configuration class for the FireFighting environment.

    """

    # --- Environment Parameters ---
    GRID_SIZE = 30
    CELL_SIZE = 30
    MAX_EPISODE_STEPS = 300
    MIN_SPAWN_DISTANCE_FROM_UAV = 6

    # --- UAV Parameters ---
    N_UAVS = 1
    UAV_VISION_RANGE = 5
    UAV_WATER_COUNT = 7
    UAV_WATER_RANGE = 4

    # --- Fire Parameters ---
    MIN_FIRES = 2
    MAX_FIRES = 9
    FIRE_HP_EACH = 2

    # --- Obstacle Parameters ---
    MIN_OBSTACLES = 2
    MAX_OBSTACLES = 5

    # --- Dense Smoke Parameters ---
    MIN_SMOKE_AREAS = 2
    MAX_SMOKE_AREAS = 7
    SMOKE_RANGE = 4
    SMOKE_IMPAIRMENT_FACTOR = 0.15

    # --- Actions: orientation index -> 2D movement vector ---
    ORIENTATION_TO_VECTOR = {
        0: (0, -1),
        1: (1, 0),
        2: (0, 1),
        3: (-1, 0)
    }

    REWARD_EXTINGUISH_FIRE  =  250
    REWARD_DOUSE_FIRE       =   50
    REWARD_FIND_FIRE        =   15

    PENALTY_STEP            =  -0.1
    PENALTY_WASTED_DOUSE_CMD=  -20
    PENALTY_WALL_COLLISION  =  -8
    PENALTY_UAV_COLLISION   = -12
    PENALTY_IMPAIRED_BY_SMOKE = -20

    # --- Rendering ---
    DOUSE_DURATION = 5

    COLOR_BACKGROUND      = (25,  35,  40)
    COLOR_GRID            = (35,  45,  50)
    COLOR_OBSTACLE        = (100, 100, 120)
    COLOR_HUD_TEXT        = (180, 190, 200)
    COLOR_AMMO_TEXT       = (255, 200, 100)
    COLOR_DOUSE           = (0,   255, 255)
    COLOR_UAV             = (0,   180, 255)
    COLOR_VISION_PULSE    = (0,   180, 255, 40)
    COLOR_FIRE_HIDDEN     = (255,  50,  50)
    COLOR_FIRE_FOUND      = (80,  200, 120)
    COLOR_HEALTH_BAR_BG   = (80,   20,  20)
    COLOR_HEALTH_BAR_FG   = (60,  200, 100)
    COLOR_FIRE_RED        = (255,   0,   0)
    COLOR_FIRE_ORANGE     = (255, 165,   0)
    COLOR_FIRE_YELLOW     = (255, 255,   0)
    COLOR_WATER           = (0,   255, 240)
    COLOR_SMOKE           = (200,   0, 255)
    COLOR_SMOKE_FIRE      = (255, 100,   0)
    COLOR_SMOKE_CORE      = (180, 190, 200, 55)
    COLOR_SMOKE_RANGE     = (150, 160, 170, 20)
    COLOR_ROCK            = (85,   95, 100)
    COLOR_ROCK_HIGHLIGHT  = (110, 120, 125)
    COLOR_ROCK_SHADOW     = (20,   25,  30)

    # --- Scoring (separate from RL rewards) ---
    SCORE_FIRE_FOUND = 5
    SCORE_FIRE_EXTINGUISHED = 10
    SCORE_FIRE_DOUSED = 7
    SCORE_TIME_DECAY = -0.02
    SCORE_CRASHED = -15
    SCORE_IMPAIRED_BY_SMOKE = -15