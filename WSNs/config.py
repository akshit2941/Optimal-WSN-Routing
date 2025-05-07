# Environment dimensions
AREA_WIDTH = 600
AREA_HEIGHT = 600

# Sensor and MC Parameters
NUM_SENSORS = 20
SENSOR_CAPACITY = 6000  # Joules
MC_CAPACITY = 1000000   # Joules
MC_SPEED = 5  # m/s
CHARGING_RATE = 20  # Joules per second
MOVEMENT_COST_PER_M = 5  # Joules per meter
THRESHOLD_RATIO = 0.2    # Time window opens when energy < 20% of capacity
CHARGING_RADIUS = 50     # Radius in meters within which MC can charge multiple nodes