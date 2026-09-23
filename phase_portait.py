import numpy as np
import matplotlib.pyplot as plt


def smooth(arr, window=15):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


cases_to_plot = [(0.75, False), (0.75, True)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for i, (level, use_nn) in enumerate(cases_to_plot):
    traj = np.load(f'results/traj_{level}_{use_nn}.npz')['trajectory']
    theta = smooth(traj[:, 0])
    theta_dot = smooth(traj[:, 1])

    ax = axes[i]
    ax.plot(theta, theta_dot, linewidth=1.3, color='tab:blue', alpha=0.9)

    ax.plot(theta[0], theta_dot[0], 'o', color='green', markersize=9,
             markeredgecolor='black', label='start', zorder=5)
    ax.plot(theta[-1], theta_dot[-1], 's', color='red', markersize=9,
             markeredgecolor='black', label='end', zorder=5)
    ax.plot(0, 0, '*', color='gold', markeredgecolor='black', markersize=18,
             label='upright equilibrium', zorder=5)
    ax.plot(np.pi, 0, 'X', color='orangered', markeredgecolor='black', markersize=13,
             label='downward equilibrium', zorder=5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.6)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.6)

    ax.set_xlim(-0.5, 4.3)
    ax.set_ylim(-3.2, 10.5)

    label = 'NN-corrected' if use_nn else 'Nominal only'
    ax.set_title(f'75% discrepancy, {label}', fontsize=13)
    ax.set_xlabel('theta [rad]')
    ax.set_ylabel('theta_dot [rad/s]')
    if i == 0:
        ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig('results/phase_portrait_75pct.png', dpi=150)
plt.show()