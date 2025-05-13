# Environment dimensions
AREA_WIDTH = 600
AREA_HEIGHT = 600

# Sensor and MC Parameters
NUM_SENSORS = 20
SENSOR_CAPACITY = 10800  # Joules (10.8 kJ)
MC_CAPACITY = 1000000   # Joules
MC_SPEED = 5  # m/s
CHARGING_RATE = 4.8  # Joules per second
MOVEMENT_COST_PER_M = 600  # Joules per meter
THRESHOLD_RATIO = 0.2    # Time window opens when energy < 20% of capacity
CHARGING_RADIUS = 50     # Radius in meters (increased from 2.8 to provide "good range")