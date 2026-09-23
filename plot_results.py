import numpy as np
import matplotlib.pyplot as plt

discrepancy_levels = [0.05, 0.15, 0.30, 0.50, 0.75]
dt = 0.02

fig, axes = plt.subplots(len(discrepancy_levels), 1, figsize=(8, 16), sharex=True)

for i, level in enumerate(discrepancy_levels):
    traj_false = np.load(f'results/traj_{level}_False.npz')['trajectory']
    traj_true = np.load(f'results/traj_{level}_True.npz')['trajectory']

    time = np.arange(len(traj_false)) * dt

    axes[i].plot(time, traj_false[:, 0], label='Nominal only (no NN)')
    axes[i].plot(time, traj_true[:, 0], label='NN-corrected')
    axes[i].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[i].set_ylabel('theta [rad]')
    axes[i].set_title(f'Discrepancy = {level}')
    axes[i].legend()

axes[-1].set_xlabel('time [s]')
plt.tight_layout()
plt.savefig('results/theta_comparison.png', dpi=150)
plt.show()



discrepancy_levels = [0.05, 0.15, 0.30, 0.50, 0.75]
rmse_values = [0.0125, 0.0240, 0.0321, 0.0770, 0.3389]

plt.figure(figsize=(6, 4))
plt.plot(discrepancy_levels, rmse_values, marker='o', linewidth=2)
plt.xlabel('Nominal model length discrepancy')
plt.ylabel('Validation RMSE')
plt.title('Residual NN prediction error vs. discrepancy severity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/rmse_vs_discrepancy.png', dpi=150)
plt.show()


final_theta_no_nn = []
final_theta_nn = []

for level in discrepancy_levels:
    traj_false = np.load(f'results/traj_{level}_False.npz')['trajectory']
    traj_true = np.load(f'results/traj_{level}_True.npz')['trajectory']
    final_theta_no_nn.append(abs(traj_false[-1, 0]))
    final_theta_nn.append(abs(traj_true[-1, 0]))

x = np.arange(len(discrepancy_levels))
width = 0.35

plt.figure(figsize=(7, 4))
plt.bar(x - width/2, final_theta_no_nn, width, label='Nominal only (no NN)')
plt.bar(x + width/2, final_theta_nn, width, label='NN-corrected')
plt.xticks(x, [f'{int(l*100)}%' for l in discrepancy_levels])
plt.xlabel('Nominal model length discrepancy')
plt.ylabel('|theta| at t=18s [rad]')
plt.title('Final angular deviation: nominal-only vs. NN-corrected')
plt.axhline(np.pi, color='red', linestyle=':', linewidth=1, label='Downward equilibrium (π)')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/final_theta_comparison.png', dpi=150)
plt.show()