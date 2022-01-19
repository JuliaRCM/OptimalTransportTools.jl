function mysolve!(x, s::QuasiNewtonOptimizer, callback!; n::Int = 0)
   
    SimpleSolvers.setInitialConditions!(s, x)

    local nmax::Int = n > 0 ? nmax = n : s.params.nmax

    s.status.i = 0
    if s.status.rgₐ ≥ s.params.atol² || n > 0 || s.params.nmin > 0
        for s.status.i = 1:nmax
            
            # update status and temporaries
            SimpleSolvers.update!(s.status)

            # apply line search
            SimpleSolvers._linesearch!(s)

            # update warmstart potentials
            callback!(s.status.x, s.F)

            # compute Gradient at new solution
            SimpleSolvers.gradient(s)(s.status.g, s.status.x)

            # compute residual
            SimpleSolvers.residual!(s.status)

            # check for convergence
            if SimpleSolvers.check_solver_converged(s.status, s.params) && s.status.i ≥ s.params.nmin && !(n > 0)
                if s.params.nwarn > 0 && s.status.i ≥ s.params.nwarn
                    println("WARNING: Quasi-Newton Optimizer took ", s.status.i, " iterations.")
                end
                break
            end

            # update Hessian
            SimpleSolvers.update!(s.H, s.status)
        end
    end
end