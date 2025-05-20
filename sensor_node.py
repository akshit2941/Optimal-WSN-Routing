import random
from config import SENSOR_CAPACITY, THRESHOLD_RATIO

class SensorNode:
    def __init__(self, node_id, x, y):
        self.id = node_id
        self.x = x
        self.y = y
        self.capacity = SENSOR_CAPACITY
        self.energy = random.uniform(0.1, 1.4) * SENSOR_CAPACITY
        self.consumption_rate = random.uniform(0.00001, 0.0001)  # Reduced to microjoules/sec range
        self.dead = False
    
    def get_location(self):
        return (self.x, self.y)

    def update_energy(self, seconds):
        if not self.dead:
            # Random energy spike for 5% of updates
            if random.random() < 0.05 and seconds > 0:
                spike_factor = random.uniform(1.5, 3.0)
                self.energy -= self.consumption_rate * seconds * spike_factor
            else:
                self.energy -= self.consumption_rate * seconds
                
            if self.energy <= 0:
                self.energy = 0
                self.dead = True

    def needs_charging(self):
        return self.energy < (THRESHOLD_RATIO * self.capacity) and not self.dead

    def charge(self, amount):
        if not self.dead:
            self.energy += amount
            if self.energy > self.capacity:
                self.energy = self.capacity