"""Provides a collection of common benchmark target functions.

These functions are typically used for testing optimization and regression
algorithms. The input `x` for each function is generally expected to be
a 1-D NumPy array representing a point in a d-dimensional space, with
values in each dimension initially scaled to the `[0,1]` range. This
input is then mapped to the function's standard domain before computation.
"""
import numpy as np


def Eggholder_2d(x):
    """Computes the 2-dimensional Eggholder function.

    This is a helper function used by the N-dimensional `Eggholder` function.
    It is known for its complex landscape with many local minima.

    Args:
        x (np.ndarray): A 1-D NumPy array of length 2, representing a point
            in 2D space (e.g., `[x0, x1]`). Values are assumed to be
            already scaled to the function's standard domain (typically [-512, 512]).

    Returns:
        float: The value of the 2D Eggholder function at point `x`.
    """
    term1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0]/2. + 47.)))
    term2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47.))))
    func = term1 + term2
    func += 959.6407
    return func

def Eggholder(x):
    """Computes the N-dimensional Eggholder function.

    The Eggholder function is a common benchmark for optimization algorithms,
    characterized by a large number of local minima, making it challenging
    to optimize. This implementation expects input `x` to be a 1-D NumPy array
    with values in the range `[0,1]` for each dimension. These values are then
    scaled to the standard Eggholder domain of `[-512, 512]` for each dimension.
    The N-dimensional function is computed by summing 2D Eggholder results
    for adjacent pairs of dimensions.

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the N-dimensional Eggholder function at point `x`.
    """
    xmin, xmax = -512, 512
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    for i in range(dim-1):
        func += Eggholder_2d(x[i:i+2])

    return func


def Himmelblau(x):
    """Computes the N-dimensional Himmelblau function.

    The Himmelblau function is often used as a benchmark for optimization.
    It has a relatively small number of local minima (typically 4 in its 2D form).
    This implementation expects input `x` to be a 1-D NumPy array with values
    in the range `[0,1]` for each dimension. These values are then scaled to
    the standard Himmelblau domain of `[-5, 5]`. The N-dimensional version
    is constructed by summing 2D Himmelblau-like terms for adjacent pairs of
    dimensions. The result is log-transformed and shifted to have a minimum of zero
    for known dimensionalities (2 to 6).

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the N-dimensional Himmelblau function at point `x`.

    Raises:
        Exception: If the dimensionality is greater than 6 and the minimum
            value for that dimensionality is not predefined.
    """
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
    """Computes the N-dimensional Rosenbrock function.

    The Rosenbrock function, also known as Rosenbrock's valley or Rosenbrock's
    banana function, is a non-convex function used as a performance test problem
    for optimization algorithms. It has a global minimum inside a long, narrow,
    parabolic-shaped flat valley.
    This implementation expects input `x` to be a 1-D NumPy array with values
    in the range `[0,1]` for each dimension. These values are then scaled to
    the domain `[-5, 10]`.

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the N-dimensional Rosenbrock function at point `x`.
    """
    xmin, xmax = -5, 10
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    for i in range(dim-1):
        func += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
    
    return func


def Rastrigin(x):
    """Computes the N-dimensional Rastrigin function.

    The Rastrigin function is a non-convex function used as a performance test
    problem for optimization algorithms. It is known for having many local
    minima, arranged in a regular grid. It has a global minimum at x_i = 0.
    This implementation expects input `x` to be a 1-D NumPy array with values
    in the range `[0,1]` for each dimension. These values are then scaled to
    the standard Rastrigin domain of `[-5.12, 5.12]`.

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the N-dimensional Rastrigin function at point `x`.
    """
    xmin, xmax = -5.12, 5.12
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    func = 10 * dim
    for i in range(dim):
        func += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    return func


def Levy(x):
    """Computes the N-dimensional Levy function.

    The Levy function is a benchmark problem for global optimization. It is
    characterized by many local minima.
    This implementation expects input `x` to be a 1-D NumPy array with values
    in the range `[0,1]` for each dimension. These values are then scaled to
    the standard Levy domain of `[-10, 10]`.

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the N-dimensional Levy function at point `x`.
    """
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
    """Computes a custom composite function.

    This function is a weighted sum of other benchmark functions defined in
    this module: Rastrigin, Eggholder, and Levy. It serves as an example of
    how more complex target functions can be constructed. The input `x` is
    passed to each component function, which assumes `x` contains values in
    `[0,1]` for each dimension before their respective domain scaling.

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the custom composite function at point `x`.
    """
    func = 0
    func += Rastrigin(x)
    func += Eggholder(x) / 6.
    func += Levy(x)
    return func