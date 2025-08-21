import numpy as np
import ResidueMinimizing

# Define the source function
def source(x, y):
    # Example: quadratic function
    return x ** 2 + y **2 - 1.0

# Define the geometric resolution
ds = 0.1  # a reasonable value

def main():
    # Minimize and get the solution
    y_solution = ResidueMinimizing.ResidueMinimize(source, ds)
    
    # Output the solution
    print("The solution y is:", y_solution.x)

if __name__ == "__main__":
    main()
