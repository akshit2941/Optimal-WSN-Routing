import numpy as np
from config import MC_CAPACITY, MOVEMENT_COST_PER_M, CHARGING_RADIUS

class MobileCharger:
    def __init__(self, base_x=300, base_y=300):
        self.x = base_x
        self.y = base_y
        self.energy = MC_CAPACITY
        self.start_pos = (base_x, base_y)

    def move_to(self, x, y):
        distance = np.linalg.norm([self.x - x, self.y - y])
        cost = distance * MOVEMENT_COST_PER_M
        if self.energy < cost:
            return False  # not enough energy to move
        self.x, self.y = x, y
        self.energy -= cost
        return True

    def charge_node(self, node):
        if self.energy <= 0 or node.dead:
            return 0
        needed = node.capacity - node.energy
        energy_to_transfer = min(needed, self.energy)
        node.charge(energy_to_transfer)
        self.energy -= energy_to_transfer
        return energy_to_transfer
        
    def charge_nodes_in_radius(self, sensors):
        """Charge all sensor nodes within the charging radius"""
        charged_nodes = []
        total_energy_transferred = 0
        
        if self.energy <= 0:
            return charged_nodes, 0
            
        # Find sensors within the charging radius that need charging
        for sensor in sensors:
            if sensor.dead:
                continue
                
            distance = np.linalg.norm([self.x - sensor.x, self.y - sensor.y])
            if distance <= CHARGING_RADIUS and sensor.energy < sensor.capacity:
                needed = sensor.capacity - sensor.energy
                # Reduce charging efficiency with distance
                efficiency = max(0.3, 1 - (distance / CHARGING_RADIUS) * 0.7)  # 30% to 100% efficiency
                max_transfer = min(needed, self.energy)
                energy_to_transfer = max_transfer * efficiency
                
                if energy_to_transfer > 0:
                    sensor.charge(energy_to_transfer)
                    self.energy -= energy_to_transfer
                    total_energy_transferred += energy_to_transfer
                    charged_nodes.append(sensor.id)
                    
                    if self.energy <= 0:
                        break
                        
        return charged_nodes, total_energy_transferred