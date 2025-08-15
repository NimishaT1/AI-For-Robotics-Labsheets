import numpy as np
import matplotlib.pyplot as plt

def traj(v, dt):
    segment_time = 2.0 / v
    steps_per_segment = int(segment_time / dt)
    headings = [0, np.pi/2, np.pi, -np.pi/2]  #turn angles
    
    x, y, theta = 0.0, 0.0, 0.0
    x_list, y_list, theta_list = [x], [y], [theta]
    
    for h in headings:
        theta = h  # Instant turn
        for a in range(steps_per_segment):
            x += v * dt * np.cos(theta)
            y += v * dt * np.sin(theta)
            x_list.append(x)
            y_list.append(y)
            theta_list.append(theta)
    
    return np.array(x_list), np.array(y_list), np.array(theta_list)

def noisy_traj(v, dt, sigma_v, sigma_omega):
    segment_time = 2.0 / v
    steps_per_segment = int(segment_time / dt)
    
    turn_angle = np.pi / 2
    turn_duration = 0.5  # seconds for turn
    steps_turn = int(turn_duration / dt)
    omega_turn = turn_angle / turn_duration  # rad/s
    
    x, y, theta = 0.0, 0.0, 0.0
    x_list, y_list, theta_list = [x], [y], [theta]
    
    for i in range(4):
       
        for a in range(steps_per_segment):
            v_noisy = v + np.random.normal(0, sigma_v)
            omega_noisy = 0 + np.random.normal(0, sigma_omega)
            x += v_noisy * dt * np.cos(theta)
            y += v_noisy * dt * np.sin(theta)
            theta += omega_noisy * dt
            x_list.append(x)
            y_list.append(y)
            theta_list.append(theta)
        
        # Turn segment
        for b in range(steps_turn):
            v_noisy = 0 + np.random.normal(0, sigma_v)
            omega_noisy = omega_turn + np.random.normal(0, sigma_omega)
            x += v_noisy * dt * np.cos(theta)
            y += v_noisy * dt * np.sin(theta)
            theta += omega_noisy * dt
            x_list.append(x)
            y_list.append(y)
            theta_list.append(theta)
    
    return np.array(x_list), np.array(y_list), np.array(theta_list)

x_ideal, y_ideal, _ = traj(1,0.1)
x_noisy, y_noisy, _ = noisy_traj(1,0.1,0.05,0.02)

# errors
min_len = min(len(x_ideal), len(x_noisy))
errors = np.sqrt((x_ideal[:min_len] - x_noisy[:min_len])**2 + 
                 (y_ideal[:min_len] - y_noisy[:min_len])**2)

# Plots
plt.figure(figsize=(5,5))
plt.plot(x_ideal, y_ideal, label="Ideal", linewidth=2)
plt.plot(x_noisy, y_noisy, label="Noisy", linestyle='--')
plt.title("Ideal vs Noisy")
plt.xlabel("X(m)")
plt.ylabel("Y(m)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

# error plot
plt.figure(figsize=(6,4))
plt.plot(np.arange(min_len)*0.1, errors, color='red')
plt.title("Euclidean Position Error")
plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.grid(True)
plt.show()
"""
the error doesn't grow linearly with time as the odometry drift is often changing and not constant.
sudden jumps occur at the turns when error in angular velocity causes large change in direction
drift accumulates over time since there is no error correction applied.
"""
