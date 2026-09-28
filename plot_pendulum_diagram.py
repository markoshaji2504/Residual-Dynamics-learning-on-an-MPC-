import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6, 7))

pivot = np.array([0.0, 0.0])
theta_illustrative = np.radians(35)
L = 1.6

rod_dir = np.array([np.sin(theta_illustrative), np.cos(theta_illustrative)])
bob = pivot + L * rod_dir
perp = np.array([np.cos(theta_illustrative), -np.sin(theta_illustrative)])

# --- rod: outlined polygon (grey fill, black edge), textbook style ---
w = 0.055
corners = np.array([
    pivot - w * perp,
    pivot + w * perp,
    bob + w * perp,
    bob - w * perp,
])
rod = patches.Polygon(corners, closed=True, facecolor='lightgray',
                       edgecolor='black', linewidth=1.2, zorder=3)
ax.add_patch(rod)

# --- vertical reference line, extends slightly below pivot (mounting post) ---
ax.plot([0, 0], [-0.15, 2.3], color='black', linewidth=1.2, zorder=1)

# --- fixed pivot marker (simple filled circle) ---
ax.add_patch(patches.Circle(pivot, 0.05, facecolor='black', zorder=4))

# --- mass (bob) ---
ax.add_patch(patches.Circle(bob, 0.22, facecolor='black', zorder=5))
ax.text(bob[0] - 0.10, bob[1] + 0.42, r'$m$', fontsize=20, zorder=6)

# --- length label ---
mid = pivot + 0.55 * (bob - pivot)
label_l = mid + 0.22 * perp
ax.text(label_l[0], label_l[1], r'$l$', fontsize=18)

# --- theta: double-headed arrow between rod and vertical, textbook style ---
arc_radius = 0.85
th1 = 90 - np.degrees(theta_illustrative)
th2 = 90
arc = patches.Arc(pivot, arc_radius * 2, arc_radius * 2, angle=0,
                   theta1=th1, theta2=th2, color='black', linewidth=1.3, zorder=3)
ax.add_patch(arc)
for ang_deg in [th1, th2]:
    ang_rad = np.radians(ang_deg)
    tip = pivot + arc_radius * np.array([np.cos(ang_rad), np.sin(ang_rad)])
    tangent = ang_rad + (np.pi / 2 if ang_deg < 90 else -np.pi / 2)
    base = tip - 0.05 * np.array([np.cos(tangent), np.sin(tangent)])
    ax.annotate('', xy=tip, xytext=base,
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.3, mutation_scale=13))
theta_mid_rad = np.radians((th1 + th2) / 2)
theta_label_pos = pivot + (arc_radius + 0.20) * np.array([np.cos(theta_mid_rad), np.sin(theta_mid_rad)])
ax.text(theta_label_pos[0], theta_label_pos[1], r'$\theta$', fontsize=18)

# --- x-y axes, closer to the pendulum ---
axis_origin = np.array([-1.1, -0.9])
ax.annotate('', xy=axis_origin + [0.5, 0], xytext=axis_origin,
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1.3))
ax.annotate('', xy=axis_origin + [0, 0.5], xytext=axis_origin,
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1.3))
ax.text(axis_origin[0] + 0.6, axis_origin[1] - 0.05, r'$x$', fontsize=15)
ax.text(axis_origin[0] - 0.05, axis_origin[1] + 0.6, r'$y$', fontsize=15)

ax.set_xlim(-1.5, 2.0)
ax.set_ylim(-1.3, 2.6)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('results/pendulum_diagram_v3.png', dpi=150, bbox_inches='tight')
plt.show()