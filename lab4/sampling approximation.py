# =================================================================
# BITS F327 : AI for Robotics
# Assignment : Sampling Approximation of Position Belief
# =================================================================
import numpy as np
import matplotlib.pyplot as plt

def apply_motion_model(particles, move_command, noise_params):
    turn, forward = move_command
    turn_noise_std = noise_params['turn_noise']
    forward_noise_std = noise_params['forward_noise']
    new_particles = []
    for p in particles:
        noise_turn = np.random.normal(turn, turn_noise_std)
        noise_forward = np.random.normal(forward, forward_noise_std)
        x_new = p['x']+noise_forward*np.cos(p['orientation']+noise_turn)
        y_new = p['y']+noise_forward*np.sin(p['orientation']+noise_turn)
        orientation_new = (p['orientation']+noise_turn)%(2*np.pi)
        new_particles.append({'x': x_new, 'y': y_new, 'orientation': orientation_new})
    return new_particles


if __name__ == '__main__':
    num_particles = 2000
    initial_pose = {'x': 0.0, 'y': 0.0, 'orientation': np.pi / 2}
    move_commands = [
        (0.0, 10.0), (-np.pi/2, 0.0), (0.0, 15.0),
        (np.pi/2, 0.0), (0.0, 10.0), (np.pi/2, 0.0), (0.0, 15.0)
    ]
    noise_params = {'turn_noise': 0.1, 'forward_noise': 0.2}
    particles = [initial_pose.copy() for _ in range(num_particles)]
    true_path = [initial_pose]
    particle_history = [particles]
    current_pose = initial_pose.copy()

    for command in move_commands:
        # Update true pose
        turn, forward = command
        current_pose['orientation'] += turn
        current_pose['x'] += forward * np.cos(current_pose['orientation'])
        current_pose['y'] += forward * np.sin(current_pose['orientation'])
        true_path.append(current_pose.copy())

        # Update particles
        particles = apply_motion_model(particles, command, noise_params)
        particle_history.append(particles)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    for step, particles_at_step in enumerate(particle_history):
        xs = [p['x'] for p in particles_at_step]
        ys = [p['y'] for p in particles_at_step]
        ax.scatter(xs, ys, alpha=0.2, s=2)

    true_xs = [p['x'] for p in true_path]
    true_ys = [p['y'] for p in true_path]
    ax.plot(true_xs, true_ys, 'k-', linewidth=2, label="True Path")
    ax.scatter(true_xs[0], true_ys[0], c="green", s=80, label="Start")  # start point

    ax.set_title("Sampling Approximation of Position Belief")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.legend()
    plt.show()
