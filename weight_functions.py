def gaussian(r, h, dim):
    w_inv = exp((r/h)**2)*pow(h, dim)*pow(math.pi, 0.5*dim)
    return 1/w_inv

def cubic_spline(r, h, dim):
    sigma = 2/3
    if(dim == 2):
        sigma = 10/(7*math.pi)
    if(dim == 3):
        sigma = 1/math.pi
    q = r/h
    w = 0
    if(0 <= g and g < 1):
        w = 1 - 1.5*q**2 + 0.75*q**3
    elif(1 <= g and g <= 2):
        w = 0.25*(2 - q)**3
    return w

def wendland(r, h, dim):
    sigma = 0.625
    if(dim == 2):
        sigma = 7/math.pi
    if(dim == 3):
        sigma = 1.3125/math.pi
    q = r/h
    w = sigma/pow(h, dim)*(1 - q)**4*(4*q + 1)
    return w

def quintic_spline(r, h, dim):
    sigma = 1/(120*h)
    if(dim == 2):
        sigma = 7/(478*h**h*math.pi)
    elif(dim == 3):
        sigma = 1/(120*h**3*math.pi)
    q = r/h
    w = 0
    if(0 <= q and q < 1):
        w = pow(3 - q, 5) - 6*pow(2 - q, 5) + 15*pow(1 - q, 5)
    if(1 <= q and q < 2):
        w = pow(3 - q, 5) - 6*pow(2 - q, 5)
    if(2 <= q and q <= 3):
         w = pow(3 - q, 5)
    return sigma*w 
