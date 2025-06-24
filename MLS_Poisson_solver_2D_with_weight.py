import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import solve
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

# --- Parameters ---
ds = 0.1
radius = 1.0

# --- 1. Boundary Nodes (evenly spaced on circle) ---
N_boundary = int(2 * np.pi * radius / ds)
theta = np.linspace(0, 2 * np.pi, N_boundary, endpoint=False)
boundary_nodes = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

# --- 2. Interior Nodes (grid points inside circle) ---
xv, yv = np.meshgrid(np.arange(-radius, radius+ds, ds), np.arange(-radius, radius+ds, ds))
grid_points = np.stack([xv.ravel(), yv.ravel()], axis=1)
dist_from_center = np.linalg.norm(grid_points, axis=1)
interior_mask = dist_from_center < radius - 1e-10
interior_nodes = grid_points[interior_mask]

# --- 3. Combine Nodes ---
all_nodes = np.vstack([boundary_nodes, interior_nodes])
N_total = all_nodes.shape[0]
N_boundary = boundary_nodes.shape[0]
N_interior = interior_nodes.shape[0]

# --- 4. Neighbor Search (within 2*ds) ---
tree = cKDTree(all_nodes)
neighbor_radius = 2 * ds
neighbors = tree.query_ball_point(interior_nodes, neighbor_radius)

# --- 5. Basis and Weight Function ---
def p(x, y):
    return np.array([1, x, y, x**2, x*y, y**2])

def weight(x, y, xj, yj, h=2*ds):
    r2 = (x - xj)**2 + (y - yj)**2
    return np.exp(-r2 / h**2)

# --- 6. Source Function ---
def source_func(x, y):
    return 1.0  # constant source

# --- 7. Assemble Laplacian Coefficients ---
A_data = []
A_row = []
A_col = []
b = np.zeros(N_interior)

lap_p = np.array([0, 0, 0, 2, 0, 2])  # Laplacian of basis

for i, (xi, yi) in enumerate(interior_nodes):
    idx_neighbors = neighbors[i]
    xj = all_nodes[idx_neighbors, 0]
    yj = all_nodes[idx_neighbors, 1]
    pj = np.stack([p(x, y) for x, y in zip(xj, yj)], axis=1)  # (6, n_neighbors)
    Wj = np.array([weight(xi, yi, x, y) for x, y in zip(xj, yj)])
    A_mat = (pj * Wj) @ pj.T
    A_inv = np.linalg.pinv(A_mat)
    for j, (xj_, yj_) in enumerate(zip(xj, yj)):
        pj_ = p(xj_, yj_)
        coeff = lap_p.T @ A_inv @ (pj_ * Wj[j])
        A_row.append(i)
        A_col.append(idx_neighbors[j])
        A_data.append(coeff)
    b[i] = source_func(xi, yi)

A = coo_matrix((A_data, (A_row, A_col)), shape=(N_interior, N_total)).tocsr()
A_interior = A[:, N_boundary:]  # only interior nodes are unknowns

# --- 8. Solve Linear System ---
u_interior = solve(A_interior.toarray(), b)

# --- 9. Collect Solution ---
u = np.zeros(N_total)
u[N_boundary:] = u_interior  # boundary nodes are zero

# --- 10. Visualization ---
plt.figure(figsize=(6,6))
plt.scatter(all_nodes[:,0], all_nodes[:,1], c=u, cmap='viridis', s=30)
plt.colorbar(label='u')
plt.gca().set_aspect('equal')
plt.title('Solution u on Unit Circle')
# --- Compute True Solution ---
x_all = all_nodes[:, 0]
y_all = all_nodes[:, 1]
u_true = ((x_all**2 + y_all**2) - 1) / 4

# --- Compute Error ---
error = u - u_true

# --- Visualization: Error ---
plt.figure(figsize=(6,6))
plt.scatter(all_nodes[:,0], all_nodes[:,1], c=error, cmap='coolwarm', s=30)
plt.colorbar(label='Error (numerical - true)')
plt.gca().set_aspect('equal')
plt.title('Solution Error on Unit Circle')
plt.show()
plt.show()
