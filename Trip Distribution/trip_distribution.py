
"""
file containing functions needed for trip distribution modeling
"""
import numpy as np
from scipy.optimize import minimize_scalar


def gravity(p, a, C, b):
    """
    Function to calculate the Balanced Gravity Model
    
    Parameters:
    ----------
    
    Parameters:
        p ((n, ) ndarray): vector of productions, length n.
        A ((n, ) ndarray): vector of attractions, lenth n.
        C ((n,n) ndarray): matrix of impedances, dim n x n.
        b (int): impedance parameter.
    ----------

    Returns:
        trips ((n,n) ndarray): trip matrix w/ columns representing
                                attractions and rows representing 
                                columns, dim n x n.
        
    """
    # output matrix (init as 0 matrix)
    trips = np.zeros((len(p), len(a)))
    
    # loop over all rows (production)
    for i in range(len(p)):
        bottomA = np.sum(a * np.power(C[i, :], (-b))) # denominator
        
        # loop over all columns (attraction)
        for j in range(len(a)):
            # calculate gravity model for trips from i to j
            topA = a[j] * np.power(C[i, j], -b)
            trips[i, j] = p[i] * topA / bottomA
    
    return trips


def balance_gravity(p, a, C, b, tolerance=0.01, maxiter=100):
    """
    Function to calculate the Balanced Gravity Model
    
    Parameters:
    ----------
        p ((n, ) ndarray): vector of productions, length n.
        a ((n, ) ndarray): vector of attractions, lenth n.
        C ((n,n) ndarray): matrix of impedances, dim n x n.
        b (int): impedance parameter.
        tolerance (float): acceptable change in trips matrix.
        maxiter (int): max number of iterations allowed.
    ----------  
    Returns:
        trips ((n,n) ndarray): trip matrix w/ columns representing
                                attractions and rows representing 
                                columns, dim n x n.
    """
    # initialize starting values
    k = 0                                   #iteration counter
    astar = a                               # starting unadjusted attractions
    trips0 = np.zeros((len(p), len(a)))     #initial T is 0's
    error = np.Inf                          # first time through, error is Infinite

    # loop through algorithm
    while(error > tolerance):
        # compute gravity model with adjusted attractions, calling gravity() function
        trips = gravity(p, astar, C, b) 

        # calculate error as the change in trips in successive iterations
        error = np.sum(np.abs(trips - trips0))

        # iterate until we reach maxiter 
        if (k > maxiter):
            break

        k += 1
        trips0 = trips
        astar = astar * a / np.sum(trips, axis=0)

    return trips

def optimal_impedance(p, a, C, observed, method, tolerance=.01):
    """
    Function to find the value of impedance variable b that minimizes 
    the difference between the observed and predicted trip matrix
    
    Parameters:
    ----------
        p ((n, ) ndarray): vector of productions, length n.
        a ((n, ) ndarray): vector of attractions, lenth n.
        C ((n,n) ndarray): matrix of impedances, dim n x n.
        observed ((n,n) ndarray): trip matrix containing observed data, dim n x n.
        tolerance (float): acceptable change in trips matrix.
        maxiter (int): max number of iterations allowed.
    ----------  
    Returns:
        b (float)): impedence value b that minimizes the error observed and predicted matrices 
    """
    def objective_function(b):
        predicted = balance_gravity(p, a, C, b, tolerance)

        #if method is root-mean-squared error
        if method == 'rmse':
            return np.sqrt(np.mean((predicted - observed) ** 2))
        
        #if method is mean absolute error
        elif method == 'mae':
            return np.mean(np.abs(predicted - observed))
        
        #if method is mean absolute percentage error 
        elif method == 'mape':
            return  np.mean(np.abs((observed - predicted) / observed)) * 100

        #if method is mean squared logarithmic error
        elif method == 'msle':
            return np.mean((np.log1p(predicted) - np.log1p(observed)) ** 2)

        else:
            raise ValueError('Invalid method')

    result = minimize_scalar(objective_function)

    return result.x