import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Equilateral triangle vertices in 2D
# State 1 at top, State 2 at bottom-left, State 3 at bottom-right
vertices = np.array([
    [0.5, np.sqrt(3)/2],  # State 1 (top)
    [0.0, 0.0],           # State 2 (bottom-left)
    [1.0, 0.0],           # State 3 (bottom-right)
])

def bary_to_cart(bary, vertices):
    """Convert barycentric coordinates to Cartesian."""
    return bary @ vertices

# The point to plot
point_bary = np.array([0.6, 0.2, 0.2])
point_cart = bary_to_cart(point_bary, vertices)

fig, ax = plt.subplots(1, 1, figsize=(7, 6.5))

# Draw triangle
triangle = plt.Polygon(vertices, fill=False, edgecolor='black', linewidth=1.5)
ax.add_patch(triangle)

# Label vertices
offset = 0.04
ax.text(vertices[0][0], vertices[0][1] + offset, 'State 1',
        ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.text(vertices[1][0] - offset, vertices[1][1] - offset, 'State 2',
        ha='center', va='top', fontsize=13, fontweight='bold')
ax.text(vertices[2][0] + offset, vertices[2][1] - offset, 'State 3',
        ha='center', va='top', fontsize=13, fontweight='bold')

# Plot the point
ax.plot(point_cart[0], point_cart[1], 'o', color='crimson', markersize=10, zorder=5)
ax.annotate(f'({point_bary[0]}, {point_bary[1]}, {point_bary[2]})',
            xy=(point_cart[0], point_cart[1]),
            xytext=(point_cart[0] + 0.08, point_cart[1] + 0.05),
            fontsize=11, color='crimson',
            arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2))

# Formatting
ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.15, 1.05)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Simplex Plot', fontsize=15, pad=15)

plt.tight_layout()
plt.savefig('triangle_plot.png', dpi=150, bbox_inches='tight')
plt.show()
