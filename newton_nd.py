"""
newton_nd.py: Use multidimensional Newton method to solve f(x) = 0
"""


def newton_nd(x0, p, u, errf = 1e-8, errDeltax = 1e-8, MaxIter = 200):
    # uses Newton Method to solve the SCALAR nonlinear system f(x)=0
    # x0 is the initial guess for Newton iteration
    # p is a structure containing all parameters needed to evaluate f( )
    # u contains values of inputs 
    # errF      = absolute equation error: how close do you want f to zero?
    # errDeltax = absolute output error:   how close do you want x?
    # note: 		declares convergence if BOTH criteria are satisfied 
    # MaxIter   = maximum number of iterations allowed
    
    k = 0 # Newton iteration index
    X = np.array([x0]).T # X stores intermediate solutions as columns

    f = evalf(x0, p, u)
    errf_k = np.linalg.norm(f, np.inf)

    Deltax = 0
    errDeltax_k = 0

    while k < MaxIter and (errf_k > errf or errDeltax_k > errDeltax):
        Jf = jacobian_fd(X[:,k], p, u)
        Deltax = np.linalg.lstsq(Jf, -f)[0]
        X = np.column_stack((X, X[:, k] + Deltax))
        k = k + 1
        f = evalf(X[:,k], p, u)
        errf_k = np.linalg.norm(f, np.inf)
        errDeltax_k = np.linalg.norm(Deltax, np.inf)

    x = X[:, k] # returning only the very last solution

    if errf_k <= errf and errDeltax_k <= errDeltax:
        converged = 1
    else:
        converged = 0

    # returning the number of iterations with ACTUAL computation
    # i.e. exclusing the given initial guess
    iterations = k-1
    
    return x, converged, errf_k, errDeltax_k, iterations, X
