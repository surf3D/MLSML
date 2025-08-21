import numpy as np
import MLSResidue
from scipy.optimize import minimize

def MLSResidue_with_ds(y):
    return MLSResidue(y, source, ds)

def ResidueMinimize(source, ds):

    N = 4  # Example value

    # Initial guess
    y0 = np.random.rand(N+1)

    # Constraint: y1 = y0 (y'(0) = 0)
    def constraint_yprime0(y):
        return y[1] - y[0]

    # Constraint: yN = 0
    def constraint_yN(y):
        return y[N]

    # Constraint: yN = y_{N-1} (y'(N) = 0), but since yN = 0, this means y_{N-1} = 0
    def constraint_yprimeN(y):
        return y[N] - y[N-1]

    constraints = [
        {'type': 'eq', 'fun': constraint_yprime0},
        {'type': 'eq', 'fun': constraint_yN},
        {'type': 'eq', 'fun': constraint_yprimeN}
    ]

    # Optional: bounds for non-negativity
    bounds = [(0, None)] * (N+1)

    result = minimize(MLSResidue_with_ds, y0, bounds=bounds, constraints=constraints)

    optimal_y = result.x

    print("Optimal y-values:", optimal_y)
    print("Minimum R:", result.fun)
    
    return optimal_y 
