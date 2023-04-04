"""
newton_nd_gcr.py: Use multidimensional Newton-GCR method to solve f(x) = 0
"""


def newton_nd_gcr(x0, p, u, errf = 1e-8, errDeltax = 1e-8, MaxIter = 200, delta = 0.1, eps = 1e-3):
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
        # Generate the initial guess for x (zero)
        x = np.zeros(f.shape[0])

        # Set the initial residual to b - Ax^0 = b
        r = -f

        # Determine the norm of the initial residual
        r_norms = []
        r_norms.append(np.linalg.norm(r, 2))
        i = 0
        q = np.empty((f.shape[0],0))
        Aq = np.empty((f.shape[0],0))
        while r_norms[i] > delta * r_norms[0]:
            # Use the residual as the first guess for the new
            # search direction and multiply by A
            q = np.column_stack((q, r))
            Aq = np.column_stack((Aq, (evalf(X[:,k] + eps*q[:, i], p, u) - f)/eps)) 
            
            # Make the new Aq vector orthogonal to the previous Aq vectors,
            # and the q vectors A^TA orthogonal to the previous q vectors
            for j in range(i-1):
                beta = np.dot(Aq[:, i], Aq[:,j])
                q[:, i] = q[:, i] - beta * q[:, j]
                Aq[:, i] = Aq[:, i] - beta * Aq[:, j]

            # Make the orthogonal Aq vector of unit length, and scale the
            # q vector so that A * q  is of unit length
            norm_Aq = np.linalg.norm(Aq[:, i], 2)
            Aq[:, i] = Aq[:, i]/norm_Aq
            q[:, i] = q[:, i]/norm_Aq

            # Determine the optimal amount to change x in the p direction
            # by projecting r onto Ap
            alpha = np.dot(r, Aq[:, i])

            # Update x and r
            x = x + alpha * q[:, i]
            r = r - alpha * Aq[:, i]

            # Save the norm of r
            r_norms.append(np.linalg.norm(r,2))
            i += 1

        Deltax = x
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
