# =================================================================
# BITS F327 : AI for Robotics
# Assignment : Monte Carlo Localization
# =================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class robot:
    def __init__(self, world_size=100.0, num_particles=1000):
        self.world_size = world_size
        self.landmarks = np.array([[20.0, 20.0], [80.0, 80.0], [20.0, 80.0],
                                   [80.0, 20.0]])
        self.num_particles = num_particles
        self.x = np.random.rand() * world_size
        self.y = np.random.rand() * world_size
        self.orientation = np.random.rand() * 2.0 * np.pi
        self.forward_noise = 0.05
        self.turn_noise = 0.05
        self.sense_noise = 5.0
        self.particles = self.create_random_particles()

    def create_random_particles(self):
        particles = []
        for _ in range(self.num_particles):
            x = np.random.rand() * self.world_size
            y = np.random.rand() * self.world_size
            orientation = np.random.rand() * 2.0 * np.pi
            particles.append([x, y, orientation])
        return np.array(particles)

    def move(self, turn, forward):
        orientation = self.orientation + turn + np.random.randn() * self.turn_noise
        orientation = orientation % (2.0 * np.pi)
        dist = forward + np.random.randn() * self.forward_noise
        self.x += np.cos(orientation) * dist
        self.y += np.sin(orientation) * dist
        self.x = self.x % self.world_size
        self.y = self.y % self.world_size
        self.orientation = orientation

        # Move particles
        new_particles = []
        for px, py, p_orient in self.particles:
            p_orient = p_orient + turn + np.random.randn() * self.turn_noise
            p_orient %= 2.0 * np.pi

            p_dist = forward + np.random.randn() * self.forward_noise
            px = (px + np.cos(p_orient) * p_dist) % self.world_size
            py = (py + np.sin(p_orient) * p_dist) % self.world_size

            new_particles.append([px, py, p_orient])
        self.particles = np.array(new_particles)

    def sense(self):
        # distances (with noise)
        measurements = []
        for lx, ly in self.landmarks:
            dx = lx - self.x
            dy = ly - self.y
            dist = np.sqrt(dx**2 + dy**2) + np.random.randn() * self.sense_noise
            measurements.append(dist)

        # weights
        weights = []
        for px, py, p_orient in self.particles:
            w = 1.0
            for i, (lx, ly) in enumerate(self.landmarks):
                dx = lx - px
                dy = ly - py
                predicted_dist = np.sqrt(dx**2 + dy**2)
                error = measurements[i] - predicted_dist
                prob = np.exp(- (error ** 2) / (2 * self.sense_noise ** 2))
                prob /= np.sqrt(2 * np.pi * self.sense_noise ** 2)
                w *= prob
            weights.append(w)
        return np.array(weights)

    def get_position(self):
        return [self.x, self.y]

def visualize(robot, step, particles, ax):
    ax.clear()
    ax.set_xlim(0, robot.world_size)
    ax.set_ylim(0, robot.world_size)
    ax.set_title(f"Step {step}")
    for lx, ly in robot.landmarks:
        ax.plot(lx, ly, "ro", markersize=8)
    ax.plot(robot.x, robot.y, "bo", markersize=8, label="Robot")
    ax.scatter(particles[:, 0], particles[:, 1], s=5, alpha=0.5, color='green', label="Particles")
    ax.legend()

if __name__ == '__main__':
    myrobot = robot()
    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(20):
        myrobot.move(0.1, 5.0)
        weights = myrobot.sense()

        # Normalize the weights
        weights /= np.sum(weights)

        # Resample particles based on weights
        indices = np.random.choice(range(myrobot.num_particles), size=myrobot.num_particles, p=weights)
        myrobot.particles = myrobot.particles[indices]

        visualize(myrobot, i + 1, myrobot.particles, ax)
        plt.pause(0.5)

    plt.show()
