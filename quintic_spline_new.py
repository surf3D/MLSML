import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

N = 8
x = np.linspace(0, 2, N)
#h = np.sin(np.pi * x / 2)
h = np.random.rand(N)         # Random h values
h[-1] = 0.0                   # Set h at x=2 to 0

# Set function value, first derivative, and second derivative at both ends to 0
bc_type = (
    ((1, 0.0), (2, 0.0)),  # At x[0]: first and second derivatives = 0
    ((1, 0.0), (2, 0.0))   # At x[-1]: first and second derivatives = 0
)

spline = make_interp_spline(x, h, k=5, bc_type=bc_type)

x_dense = np.linspace(0, 2, 200)

fig, ax = plt.subplots(4, 1, figsize=(8, 16), sharex=True)

ax[0].plot(x_dense, spline(x_dense), label='Quintic Spline h(x)')
ax[0].plot(x, h, 'o', label='Knots')
ax[0].set_ylabel('h(x)')
ax[0].set_title('Quintic Spline and Its Derivatives (with boundary conditions)')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(x_dense, spline(x_dense, 1), label="First Derivative h'(x)", color='orange')
ax[1].set_ylabel("h'(x)")
ax[1].legend()
ax[1].grid(True)

ax[2].plot(x_dense, spline(x_dense, 2), label="Second Derivative h''(x)", color='green')
ax[2].set_ylabel("h''(x)")
ax[2].legend()
ax[2].grid(True)

ax[3].plot(x_dense, spline(x_dense, 3), label="Third Derivative h'''(x)", color='red')
ax[3].set_xlabel('x')
ax[3].set_ylabel("h'''(x)")
ax[3].legend()
ax[3].grid(True)

plt.tight_layout()
plt.show()
