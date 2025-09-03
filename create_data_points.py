import numpy as np

def create_data_points(ds, radius=1.0):
    N_boundary = int(2 * np.pi * radius / ds)
    theta = np.linspace(0, 2 * np.pi, N_boundary, endpoint=False)
    boundary_nodes = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    xv, yv = np.meshgrid(np.arange(-radius, radius+ds, ds), np.arange(-radius, radius+ds, ds))
    grid_points = np.stack([xv.ravel(), yv.ravel()], axis=1)
    dist_from_center = np.linalg.norm(grid_points, axis=1)
    interior_mask = dist_from_center < radius - 1e-10
    interior_nodes = grid_points[interior_mask]

    all_nodes = np.vstack([boundary_nodes, interior_nodes])
    return all_nodes, boundary_nodes, interior_nodes
