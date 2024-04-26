import numpy as np


def Eggholder_2d(x):
    term1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0]/2. + 47.)))
    term2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47.))))
    func = term1 + term2
    return func

def Eggholder(x):
    xmin, xmax = -512, 512
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    for i in range(dim-1):
        func += Eggholder_2d(x[i:i+2])

    return func


def Himmelblau(x):
    xmin, xmax = -5, 5
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    for i in range(dim-1):
        func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2
    func += 1
    func = np.log(func)

    if dim == 2:
        pass
    elif dim == 3:
        func -= 0.265331837897597
    elif dim == 4:
        func -= 1.7010318616354436
    elif dim == 5:
        func -= 2.3001107745553155
    elif dim == 6:
        func -= 2.8576426513378994
    else:
        raise Exception("We don't know the minimum value for Himmelblau in this number of dimensions.")

    return func


def Rosenbrock(x):
    xmin, xmax = -5, 10
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    for i in range(dim-1):
        func += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
    
    return func


def Rastrigin(x):
    xmin, xmax = -5.12, 5.12
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    func = 10 * dim
    for i in range(dim):
        func += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    return func


def Levy(x):
    xmin, xmax = -10, 10
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    w = []
    for i in range(dim):
        w.append(1 + (x[i]-1)/4)
    term1 = (np.sin(np.pi * w[0]))**2
    term_sum = 0
    for i in range(dim-1):
        term_sum += ((w[i] - 1)**2) * (1 + 10 * (np.sin(np.pi * w[i] + 1))**2)
    
    term_end = ((w[dim-1] - 1)**2) * (1 + (np.sin(2 * np.pi * w[dim-1])**2))
    
    func = term1 + term_sum + term_end

    return func



def Custom(x):
    func = 0
    func += Rastrigin(x)
    func += Eggholder(x) / 6.
    func += Levy(x)
    return func