"""
sinkhorn_dvg(α, β, K, a₀, b₀, d₁₀, d₂₀, SP, caches)
"""
function sinkhorn_dvg(  α::Vector{T},
                        β::Vector{V},
                        a₀, b₀,
                        d₁₀, d₂₀,
                        K,
                        SP, caches   ) where {T,V}

    VC = caches.VC

    VC[:a,V] .= a₀
    VC[:b,V] .= b₀
    if SP.debias
        VC[:d₁,V] .= d₁₀
        VC[:d₂,V] .= d₂₀
    end
    
    for l in 1:SP.L
        mul!(VC[:t1,V], K, VC[:b,V])
        VC[:a,V] .= α ./ VC[:t1,V]
        mul!(VC[:t1,V], K', VC[:a,V])
        VC[:b,V] .= β ./ VC[:t1,V]

        if SP.debias # && (l%2==0 || l==SP.L)
            mul!(VC[:t1,V], K, VC[:d₁,V])
            VC[:d₁,V] .= sqrt.( α .* VC[:d₁,V] ./ VC[:t1,V] )
            mul!(VC[:t1,V], K, VC[:d₂,V])
            VC[:d₂,V] .= sqrt.( β .* VC[:d₂,V] ./ VC[:t1,V] )
        end
    end

    S_ε = 0

    if SP.debias
        for i in eachindex(α)
            S_ε += α[i] * (_safe_log( VC[:a,V][i] ) - _safe_log( VC[:d₁,V][i] )) +
                    β[i] * (_safe_log( VC[:b,V][i] ) - _safe_log( VC[:d₂,V][i] ))
        end
    else
        for i in eachindex(α)
            S_ε += α[i] * _safe_log( VC[:a,V][i] ) +
                    β[i] * _safe_log( VC[:b,V][i] )
        end
    end

    return SP.ε * S_ε
end


function sinkhorn_barycenter(  λ::AbstractVector{T}, α::AbstractVector{AT},
                                b_₀, d₀,
                                K, SP, caches,
                                ) where {T, V, AT <: AbstractArray{V}}

    VC = caches.VC
    VVC = caches.VVC

    for s in 1:VVC.S
        VVC[:b,T][s] .= b_₀[s]
    end
    if SP.debias
        VC[:d,T] .= d₀
    end

    for l in 1:SP.L
        for s in 1:VVC.S
            mul!(VC[:t1,V], K, VVC[:b,V][s])
            VC[:t1,V] .= α[s] ./ VC[:t1,V]
            mul!(VVC[:φ,T][s], K', VC[:t1,V])
        end

        SP.debias ? VC[:μ,T] .= VC[:d,T] : VC[:μ,T] .= 1

        for s in 1:VC.S
            VC[:μ,T] .*= VVC[:φ,T][s] .^ λ[s]
        end

        for s in 1:VC.S
            VVC[:b,T][s] .= VC[:μ,T] ./ VVC[:φ,T][s]
        end

        if SP.debias
            mul!(VC[:t1,V], K, VC[:d,T])
            VC[:d,T] .= sqrt.( VC[:d,T] .* VC[:μ,T] ./ VC[:t1,T] )
        end
    end

    if SP.update_potentials
        for s in 1:VVC.S
            b_₀[s] .= ForwardDiff.value.(VVC[:b,T][s])
        end
        if SP.debias
            d₀ .= ForwardDiff.value.(VC[:d,T])
        end
    end

    return copy(VC[:μ,T])
end