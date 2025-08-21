import numpy as np
import mls_poisson_solution
import Poisson_circle_Green_function
import create_data_points
from scipy.interpolate import make_interp_spline

def weight_function(y, r):
    # length of y (also number of knots)
    num_knots = len(y) + 1
    x_knots = np.linspace(0, 2, num_knots)
 
    # Set up boundary conditions:
    # (order, value): (0, value) means function value, (1, value) means first derivative
    bc_type = [(1, 0.0), (0, 0.0), (1, 0.0)]  # (h'(0) = 0, h[2] = 0, h'(2) = 0)
    # weight function definition
    weight = make_interp_spline(r, y, k=5, bc_type=bc_type)
    return weight

def MLSResidue(y, source, ds):
    # creation of the data points
    all_nodes, boundary_nodes, interior_nodes = create_data_points(ds)
    # get the MLS solution
    u_mls = mls_poisson_solution.mls_poisson_solution(all_nodes, boundary_nodes, interior_nodes, \
                                                      source, weight_function, ds)
    u_ex  = u = Poisson_circle_Green_function.poisson_solution_on_disk_nodes(source, all_nodes)
    error = u_mls - u_ex
    L2_error = np.sqrt(np.sum(error**2))
    return L2_error