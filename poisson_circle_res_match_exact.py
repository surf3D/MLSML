import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# 1. Construct a unit circle and a grid of size 0.1
grid_size = 0.25
x_grid = np.arange(-1.0, 1.0 + grid_size/2, grid_size)
y_grid = np.arange(-1.0, 1.0 + grid_size/2, grid_size)
X, Y = np.meshgrid(x_grid, y_grid)
points = np.column_stack((X.flatten(), Y.flatten()))

# 2. Identify interior and boundary points
interior_mask = X.flatten()**2 + Y.flatten()**2 < 1.0
interior_points = points[interior_mask]
n_interior = len(interior_points)
print(f"Number of interior points: {n_interior}")

# Generate boundary points
theta = np.linspace(0, 2*np.pi, int(2*np.pi/grid_size) + 1)[:-1]  # Remove last point to avoid duplication
boundary_points = np.column_stack((np.cos(theta), np.sin(theta)))
n_boundary = len(boundary_points)
print(f"Number of boundary points: {n_boundary}")

# Combine all points
all_points = np.vstack((interior_points, boundary_points))
n_total = len(all_points)
print(f"Total number of points: {n_total}")

# 3. For each interior point, find neighbors within distance 0.20001
max_distance = 2*grid_size + 0.00001
neighbors = []
for i in range(n_interior):
    point = interior_points[i]
    # Calculate distances to all points
    distances = np.sqrt(np.sum((all_points - point)**2, axis=1))
    # Find indices of neighbors
    neighbor_indices = np.where(distances < max_distance)[0]
    neighbors.append(neighbor_indices)
    
# 4 & 5. Set up least squares to fit a quadratic polynomial for each interior point
# and form a linear system using the condition 2A + 2C = 1

# Create sparse matrix for the linear system
A = lil_matrix((n_interior, n_total))
b = np.zeros(n_interior)

for i in range(n_interior):
    point = interior_points[i]
    neighbor_indices = neighbors[i]
    neighbor_points = all_points[neighbor_indices]
    
    # Build the design matrix for least squares
    X_ls = np.ones((len(neighbor_indices), 6))
    for j, (x, y) in enumerate(neighbor_points):
        X_ls[j, 0] = x**2      # A coefficient
        X_ls[j, 1] = x*y       # B coefficient
        X_ls[j, 2] = y**2      # C coefficient
        X_ls[j, 3] = x         # D coefficient
        X_ls[j, 4] = y         # E coefficient
        X_ls[j, 5] = 1.0       # F coefficient
    
    # Calculate the least squares matrix (X^T X)^-1 X^T
    XTX_inv_XT = np.linalg.pinv(X_ls)
    
    # Row 0 corresponds to A, row 2 corresponds to C
    # The condition 2A + 2C = 1 means we need coefficients of U such that
    # 2*(row 0 of XTX_inv_XT) + 2*(row 2 of XTX_inv_XT) = right-hand side
    coeffs = 2 * XTX_inv_XT[0, :] + 2 * XTX_inv_XT[2, :]
    
    # Fill the sparse matrix row for this interior point
    for j, neighbor_idx in enumerate(neighbor_indices):
        A[i, neighbor_idx] = coeffs[j]
    
    # Right-hand side is 1
    b[i] = 1.0

# Convert to CSR format for efficient solving
A = A.tocsr()

# 6. Solve the system with boundary conditions U = 0
# First, move the boundary conditions to the right-hand side
for i in range(n_interior):
    for j in range(n_boundary):
        idx = n_interior + j
        b[i] -= A[i, idx] * 0  # U = 0 at boundary points
        A[i, idx] = 0

# Only keep the part of the matrix corresponding to interior points
A_interior = A[:, :n_interior]

# Solve the system
u_interior = spsolve(A_interior, b)

# 7. Construct the full solution
u = np.zeros(n_total)
u[:n_interior] = u_interior
# u[n_interior:] = 0  # Boundary points are already 0

# 8. Compare with exact solution S = (x^2 + y^2 - 1)/4
exact_solution = (all_points[:, 0]**2 + all_points[:, 1]**2 - 1) / 4
error = u - exact_solution[:n_total]
max_error = np.max(np.abs(error))
rms_error = np.sqrt(np.mean(error**2))

print(f"Maximum error: {max_error}")
print(f"RMS error: {rms_error}")

# Calculate residue of the linear equation for each interior point
residue = np.zeros(n_interior)
for i in range(n_interior):
    neighbor_indices = neighbors[i]
    residue[i] = np.sum(A[i, :].toarray()[0] * u) - b[i]

max_residue = np.max(np.abs(residue))
rms_residue = np.sqrt(np.mean(residue**2))

print(f"Maximum residue: {max_residue}")
print(f"RMS residue: {rms_residue}")

# 9. Plot the results
plt.figure(figsize=(15, 5))

# Plot the interior and boundary points
plt.subplot(1, 3, 1)
plt.scatter(interior_points[:, 0], interior_points[:, 1], c='blue', s=10, label='Interior')
plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='red', s=10, label='Boundary')
plt.axis('equal')
plt.title('Interior and Boundary Points')
plt.legend()

# Plot the numerical solution
plt.subplot(1, 3, 2)
sc = plt.scatter(all_points[:, 0], all_points[:, 1], c=u, cmap='viridis')
plt.colorbar(sc, label='Numerical Solution')
plt.axis('equal')
plt.title('Numerical Solution')

# Plot the error
plt.subplot(1, 3, 3)
sc = plt.scatter(all_points[:, 0], all_points[:, 1], c=np.abs(error), cmap='hot')
plt.colorbar(sc, label='Absolute Error')
plt.axis('equal')
plt.title('Error')

# Add a new figure to plot the residue
plt.figure(figsize=(10, 5))
sc = plt.scatter(interior_points[:, 0], interior_points[:, 1], c=np.abs(residue), cmap='plasma')
plt.colorbar(sc, label='Absolute Residue')
plt.axis('equal')
plt.title('Residue of Linear Equation (|Au - b|)')

plt.tight_layout()
plt.show()

# Create visualizations
create_visualizations(interior_points, boundary_points, grid_size, U, exact_values, errors, cross_section)

def create_visualizations(interior_points, boundary_points, grid_size, U, exact_values, errors, cross_section):
    """Create visualizations of the numerical solution, exact solution, and error."""
    
    # Create a regular grid for visualization
    x_grid = np.arange(-1, 1.01, grid_size/2)
    y_grid = np.arange(-1, 1.01, grid_size/2)
    
    # Ensure x=0 and y=0 are exactly in the grid
    if 0 not in x_grid:
        x_grid = np.sort(np.append(x_grid, 0))
    if 0 not in y_grid:
        y_grid = np.sort(np.append(y_grid, 0))
    
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Create mapping of interior points to their values
    point_dict_numerical = {(round(x, 10), round(y, 10)): val for (x, y), val in zip(interior_points, U)}
    point_dict_exact = {(round(x, 10), round(y, 10)): val for (x, y), val in zip(interior_points, exact_values)}
    point_dict_error = {(round(x, 10), round(y, 10)): val for (x, y), val in zip(interior_points, errors)}
    
    # Initialize matrices for visualization
    numerical_matrix = np.full_like(xx, np.nan)
    exact_matrix = np.full_like(xx, np.nan)
    error_matrix = np.full_like(xx, np.nan)
    
    # Fill matrices
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            x, y = x_grid[i], y_grid[j]
            key = (round(x, 10), round(y, 10))
            r_squared = x**2 + y**2
            
            # Fill exact solution for all points in grid
            if r_squared <= 1:
                exact_matrix[j, i] = (r_squared - 1) / 4
                
                # For numerical and error, use the computed values if available
                if key in point_dict_numerical:
                    numerical_matrix[j, i] = point_dict_numerical[key]
                    error_matrix[j, i] = point_dict_error[key]
                elif r_squared > 0.99:  # Near boundary
                    numerical_matrix[j, i] = 0  # Boundary condition
                    error_matrix[j, i] = 0      # No error at boundary
    
    # Use interpolation to fill gaps
    from scipy.interpolate import griddata
    
    # Prepare points and values for interpolation
    points = interior_points
    values_numerical = U
    values_error = errors
    
    # Grid points where we want to interpolate
    grid_points = []
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            if np.isnan(numerical_matrix[j, i]) and xx[j, i]**2 + yy[j, i]**2 < 1:
                grid_points.append((x_grid[i], y_grid[j]))
    
    if grid_points:
        # Convert to numpy array
        grid_points = np.array(grid_points)
        
        # Interpolate
        interpolated_numerical = griddata(points, values_numerical, grid_points, method='cubic')
        interpolated_error = griddata(points, values_error, grid_points, method='cubic')
        
        # Fill in the interpolated values
        for k, (x, y) in enumerate(grid_points):
            i = np.where(x_grid == x)[0][0]
            j = np.where(y_grid == y)[0][0]
            numerical_matrix[j, i] = interpolated_numerical[k]
            error_matrix[j, i] = interpolated_error[k]
    
    # Create figure with subplots
    plt.figure(figsize=(15, 12))
    
    # 1. Numerical Solution
    plt.subplot(221)
    
    # Create a circular mask
    mask = xx**2 + yy**2 > 1
    numerical_masked = np.ma.array(numerical_matrix, mask=mask)
    
    contour1 = plt.contourf(xx, yy, numerical_masked, 20, cmap='viridis')
    plt.colorbar(contour1, label='Value')
    plt.title('Numerical Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    # Draw the unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5)
    
    # Add coordinate axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 2. Exact Solution
    plt.subplot(222)
    
    # Mask points outside the circle
    exact_masked = np.ma.array(exact_matrix, mask=mask)
    
    contour2 = plt.contourf(xx, yy, exact_masked, 20, cmap='viridis')
    plt.colorbar(contour2, label='Value')
    plt.title('Exact Solution: (x² + y² - 1)/4')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    # Draw the unit circle
    plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5)
    
    # Add coordinate axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 3. Error
    plt.subplot(223)
    
    # Mask points outside the circle
    error_masked = np.ma.array(error_matrix, mask=mask)
    
    contour3 = plt.contourf(xx, yy, error_masked, 20, cmap='plasma')
    plt.colorbar(contour3, label='Error')
    plt.title('Error: |Numerical - Exact|')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    # Draw the unit circle
    plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5)
    
    # Add coordinate axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 4. Cross-section along y=0
    plt.subplot(224)
    
    # Extract data from cross_section
    x_vals = [p[0] for p in cross_section]
    num_vals = [p[1] for p in cross_section]
    exact_vals = [p[2] for p in cross_section]
    error_vals = [p[3] for p in cross_section]
    
    plt.plot(x_vals, num_vals, 'b-o', label='Numerical')
    plt.plot(x_vals, exact_vals, 'r--', label='Exact')
    plt.plot(x_vals, error_vals, 'g-', label='Error')
    plt.grid(True)
    plt.title('Cross-section along y=0')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('poisson_solution_comparison.png', dpi=150)
    print("Visualization saved as 'poisson_solution_comparison.png'")
    
    # Create 3D surface plots
    plt.figure(figsize=(15, 5))
    
    # 1. Numerical Solution 3D
    ax1 = plt.subplot(131, projection='3d')
    surf1 = ax1.plot_surface(xx, yy, numerical_masked, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_title('Numerical Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    
    # 2. Exact Solution 3D
    ax2 = plt.subplot(132, projection='3d')
    surf2 = ax2.plot_surface(xx, yy, exact_masked, cmap='viridis', edgecolor='none', alpha=0.8)
    ax2.set_title('Exact Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    
    # 3. Error 3D
    ax3 = plt.subplot(133, projection='3d')
    surf3 = ax3.plot_surface(xx, yy, error_masked, cmap='plasma', edgecolor='none', alpha=0.8)
    ax3.set_title('Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('Error')
    
    plt.tight_layout()
    plt.savefig('poisson_solution_3d.png', dpi=150)
    print("3D visualization saved as 'poisson_solution_3d.png'")
    
    # Create symmetry check visualization
    plt.figure(figsize=(12, 10))
    
    # 1. Heat map of numerical solution
    plt.subplot(221)
    contour = plt.contourf(xx, yy, numerical_masked, 20, cmap='viridis')
    plt.colorbar(contour, label='Value')
    plt.title('Numerical Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    # Draw the unit circle
    plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5)
    
    # Add coordinate axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.7)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.7)
    
    # 2. Cross-section along x-axis
    plt.subplot(222)
    
    # Extract values along x-axis (y=0)
    plt.plot(x_vals, num_vals, 'b-o', label='Numerical')
    plt.plot([-x for x in reversed(x_vals)], num_vals, 'r--', label='Mirrored')
    plt.grid(True)
    plt.title('Symmetry across y-axis (x → -x)')
    plt.xlabel('x')
    plt.ylabel('u(x,0)')
    plt.legend()
    
    # 3. Cross-section along y-axis
    plt.subplot(223)
    
    # Find points along y-axis (x=0)
    y_axis_data = []
    for i, (x, y) in enumerate(interior_points):
        if abs(x) < 1e-10:
            y_axis_data.append((y, U[i]))
    
    # Sort by y-coordinate
    y_axis_data.sort()
    
    if y_axis_data:
        y_vals = [p[0] for p in y_axis_data]
        u_vals = [p[1] for p in y_axis_data]
        
        plt.plot(y_vals, u_vals, 'b-o', label='Numerical')
        plt.plot([-y for y in reversed(y_vals)], u_vals, 'r--', label='Mirrored')
        plt.grid(True)
        plt.title('Symmetry across x-axis (y → -y)')
        plt.xlabel('y')
        plt.ylabel('u(0,y)')
        plt.legend()
    
    # 4. Diagonal cross-section
    plt.subplot(224)
    
    # Find points along diagonal (x=y)
    diag_data = []
    for i, (x, y) in enumerate(interior_points):
        if abs(x - y) < 1e-10:
            diag_data.append((x, U[i]))
    
    # Sort by x-coordinate
    diag_data.sort()
    
    if diag_data:
        diag_x = [p[0] for p in diag_data]
        diag_u = [p[1] for p in diag_data]
        
        # Diagonal points and their mirror across the origin
        plt.plot(diag_x, diag_u, 'b-o', label='x=y')
        
        # Find points along anti-diagonal (x=-y)
        anti_diag_data = []
        for i, (x, y) in enumerate(interior_points):
            if abs(x + y) < 1e-10:
                anti_diag_data.append((x, U[i]))
        
        # Sort by x-coordinate
        anti_diag_data.sort()
        
        if anti_diag_data:
            anti_x = [p[0] for p in anti_diag_data]
            anti_u = [p[1] for p in anti_diag_data]
            plt.plot(anti_x, anti_u, 'r--', label='x=-y')
        
        plt.grid(True)
        plt.title('Diagonals x=y and x=-y')
        plt.xlabel('x')
        plt.ylabel('u(x,±x)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('poisson_symmetry_check.png', dpi=150)
    print("Symmetry check visualization saved as 'poisson_symmetry_check.png'")
    
    plt.show()
