# MLSML
Machine Learning through Moving Least Squared Mesh Less solver.
Solve Poisson equation on a unit circle with MLS interpolation
Base functions: {x^2, y^2, xy, x, y, 1} for third order Taylor expansion, can have other choices.
Weight function: currently Gaussian, to be improved by ML for optimized accuracy.
Point distribution: grid points inside unit circle, boundary points on circle.
Resolution is temporarily set to 0.1.
search length: (2*resolution).
Source: 1.0, can be changed. The chosen base function provides maching precision with numerical solution.
