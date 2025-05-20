# Environment dimensions
AREA_WIDTH = 600        # Width of the simulation area in meters
AREA_HEIGHT = 600       # Height of the simulation area in meters

# Sensor and MC Parameters
NUM_SENSORS = 600               # Total number of sensors deployed
SENSOR_CAPACITY = 10800         # Maximum energy capacity of each sensor in Joules (10.8 kJ)
MC_CAPACITY = 1000000           # Maximum energy capacity of the mobile charger (MC) in Joules
MC_SPEED = 3                    # Speed of the mobile charger in meters per second
CHARGING_RATE = 3.6             # Rate at which MC charges sensors in Joules per second
MOVEMENT_COST_PER_M = 60       # Energy cost for MC to move 1 meter in Joules
THRESHOLD_RATIO = 0.2           # Threshold ratio; time window opens when sensor energy < 20% of capacity
CHARGING_RADIUS = 60            # Effective charging radius of MC in meters

# Simulation Parameters
CRITICAL_ENERGY_RATIO = 0.03        # Ratio below which sensor is considered in critical energy state (3%)
CHARGING_INNER_ZONE = 0.4           # Inner zone boundary as a fraction of charging radius (40%)
CHARGING_MIDDLE_ZONE = 0.7          # Middle zone boundary as a fraction of charging radius (70%)
INNER_EFFICIENCY = 0.75             # Charging efficiency in the inner zone (75%)
MIDDLE_EFFICIENCY = 0.55            # Charging efficiency in the middle zone (55%)
OUTER_EFFICIENCY = 0.35             # Charging efficiency in the outer zone (35%)
GRID_SEARCH_SIZE = 30               # Grid size for searching optimal charging positions
CHARGING_SEARCH_EXPANSION = 1.2     # Expansion factor for charging search area
MIN_PRIORITY_THRESHOLD = 0.15       # Minimum priority threshold for scheduling charging (15%)
MIN_SENSORS_FOR_POSITION = 3        # Minimum number of sensors required to consider a charging position
SCORE_IMPROVEMENT_FACTOR = 1.5      # Factor by which score must improve to update charging position
RETURN_THRESHOLD_RATIO = 0.15       # Ratio at which MC should return to base for recharge (15%)
CHARGE_ENERGY_ESTIMATE = 500        # Estimated energy required for a single charging operation in Joules