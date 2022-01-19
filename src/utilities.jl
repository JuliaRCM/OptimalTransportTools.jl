mutable struct SinkhornParameters
    L::Int
    ε::Real

    averaged_updates::Bool
    debias::Bool

    update_potentials::Bool
end

SinkhornParameters(L) = SinkhornParameters(L, 5e-3, false, true, false)
SinkhornParameters(L, ε) = SinkhornParameters(L, ε, false, true, false)

function _safe_log!(log_x, x)
    for i in eachindex(x)
        x[i] <= 0 ? log_x[i] = -1e4 : log_x[i] = log(x[i])
    end
end

function _safe_log(x::AbstractArray)
    log_x = zero(x)
    for i in eachindex(x)
        x[i] <= 0 ? log_x[i] = -1e4 : log_x[i] = log(x[i])
    end
    log_x
end

function _safe_log(x::Real)
    if x <= 0
        return -1e4
    else
        return log(x)
    end
end