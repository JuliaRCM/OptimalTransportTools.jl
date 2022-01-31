function mysolve!(x, s::Optimizer, callback)
    SimpleSolvers.initialize!(s, x)

    while !SimpleSolvers.meets_stopping_criteria(SimpleSolvers.status(s), SimpleSolvers.config(s))
        SimpleSolvers.next_iteration!(SimpleSolvers.status(s))
        SimpleSolvers.solver_step!(s)
        # residual!(status(s))

        callback( SimpleSolvers.solution(SimpleSolvers.status(s)) )
    end

    SimpleSolvers.warn_iteration_number(SimpleSolvers.status(s), SimpleSolvers.config(s))

    copyto!(x, SimpleSolvers.solution(SimpleSolvers.status(s)))

    return x
end
