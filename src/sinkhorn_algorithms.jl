struct Sinkhorn{T, log_domain, averaged_updates, debias}
    L::Int
    ε::T
    tol::T

    function Sinkhorn(ε::T;
                      L = 128,
                      tol::T = 0.0,
                      log_domain::Bool = false,
                      averaged_updates::Bool = false,
                      debias::Bool = true) where {T}
        new{T, log_domain, averaged_updates, debias}(ε, L, tol)
    end
end

const SinkhornRealDomain{T, averaged_updates, debias} where {T, averaged_updates, debias} = Sinkhorn{T, false, averaged_updates, debias} 
const SinkhornLogDomain{T, averaged_updates, debias} where {T, averaged_updates, debias} = Sinkhorn{T, true, averaged_updates, debias}

hasaveragedupdates(::Sinkhorn{T, ld, au}) where {T, ld, au} = au
hasdebias(::Sinkhorn{T, ld, au, db}) where {T, ld, au, db} = db