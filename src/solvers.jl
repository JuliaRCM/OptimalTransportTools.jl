function solve_callback!(x, opt::SS.Optimizer, callback)
    SS.initialize!(opt, x) 

    while !SS.meets_stopping_criteria(opt)
        SS.next_iteration!(result(opt))
        SS.solver_step!(x, state(opt))
        SS.update!(opt, x)

        callback(opt)
    end

    SS.warn_iteration_number(status(opt), config(opt))

    return x
end
