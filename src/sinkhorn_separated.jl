"""
applies a ↦ b = Ka
"""
function  apply_K_sep!(b::AbstractMatrix, a, k, tmp)
    mul!(tmp, k[1], a)       # :t[1] = k[1] * :b
    mul!(b, tmp, k[2])
end

"""
applies a ↦ b = Ka
"""
@inline function  apply_K_sep!(b::AbstractVector, a, K, tmp)
    mul!(b, K, a)
end

"""
wasserstein_distance_separated(p, β, a₀, b₀, d₁₀, d₂₀, k, SP, caches)
p: right marginal as nxn matrix
β: left marginal as nxn matrix
k: e^(c/ε) with cost c as dx(nxn) vector of matrices

Only differentiable by α

TODO: higher dimensions than d=2, number of grid points different in dimensions/marginals
"""

function sinkhorn_dvg_separated(α::AbstractArray{T}, β::AbstractArray{V},
                                a₀, b₀, d₁₀, d₂₀,
                                k, SP, caches
                                ) where {T, V}

    MC = caches.MC

    MC[:α,V] .= ForwardDiff.value.(α) 

    MC[:a,V] .= a₀;   MC[:b,V] .= b₀
    if SP.debias
        MC[:d₁,V] .= d₁₀; MC[:d₂,V] .= d₂₀
    end
   
    for l in 1:SP.L
        if SP.averaged_updates
            apply_K_sep!(MC[:t2,V], MC[:b,V], k, MC[:t1,V])
            MC[:a₊,V] .= MC[:α,V] ./ MC[:t2,V]
            apply_K_sep!(MC[:t2,V], MC[:a,V], k, MC[:t1,V])
            MC[:b₊,V] .= β ./ MC[:t2,V]
        
            MC[:b,V] .= sqrt.(MC[:b,V] .* MC[:b₊,V])
            MC[:a,V] .= sqrt.(MC[:a,V] .* MC[:a₊,V])            
        else
            apply_K_sep!(MC[:t2,V], MC[:b,V], k, MC[:t1,V])
            MC[:a,V] .= MC[:α,V] ./ MC[:t2,V]
            apply_K_sep!(MC[:t2,V], MC[:a,V], k, MC[:t1,V])
            MC[:b,V] .= β ./ MC[:t2,V]
        end

        if SP.debias #&& ( l%2==0 || l==SP.L )
            if SP.averaged_updates
                apply_K_sep!(MC[:t2,V], MC[:d₁,V], k, MC[:t1,V])
                MC[:d₁₊,V] .= sqrt.( MC[:d₁,V] .* MC[:α,V] ./ MC[:t2,V] )
                apply_K_sep!(MC[:t2,V], MC[:d₂,V], k, MC[:t1,V])
                MC[:d₂₊,V] .= sqrt.( MC[:d₂,V] .* β ./ MC[:t2,V] )

                MC[:d₁,V] .= sqrt.(MC[:d₁,V] .* MC[:d₁₊,V])
                MC[:d₂,V] .= sqrt.(MC[:d₂,V] .* MC[:d₂₊,V]) 
            else
                apply_K_sep!(MC[:t2,V], MC[:d₁,V], k, MC[:t1,V])
                MC[:d₁,V] .= sqrt.( MC[:d₁,V] .* MC[:α,V] ./ MC[:t2,V] )
                apply_K_sep!(MC[:t2,V], MC[:d₂,V], k, MC[:t1,V])
                MC[:d₂,V] .= sqrt.( MC[:d₂,V] .* β ./ MC[:t2,V] )
            end
        end
    end

    if SP.debias
        for i in eachindex(MC[:t2,T])
                MC[:t2,T][i] = α[i] * (_safe_log( MC[:a,V][i] ) - _safe_log( MC[:d₁,V][i] )) +
                               β[i] * (_safe_log( MC[:b,V][i] ) - _safe_log( MC[:d₂,V][i] ))
        end
    else
        for i in eachindex(MC[:t2,T])
                MC[:t2,T][i] = α[i] * _safe_log( MC[:a,V][i] ) + β[i] * _safe_log( MC[:b,V][i] )
        end
    end

    S_ε = SP.ε * sum( MC[:t2,T] )

    if SP.update_potentials
        a₀ .= MC[:a,V]
        b₀ .= MC[:b,V]
        if SP.debias
            d₁₀ .= MC[:d₁,V]
            d₂₀ .= MC[:d₂,V]
        end
    end
    
    return S_ε
end

function sinkhorn_dvg_separated(α::AbstractArray{T}, β::AbstractArray{V},
                                k, SP, caches
                                ) where {T, V}
    sinkhorn_dvg_separated( α, β,
                            ones(V, size(α)), ones(V, size(β)),
                            ones(V, size(α)), ones(V, size(β)),
                            k, SP, caches
                            )
end


"""
wasserstein_barycenter_separated!(α, β, k, ε, L)
w: un-normalized weights as s vector
α: input marginals as sx(nxn) vector of matrices
k: Gibbs factor as a vector of dx(nxn) matrices
ε: entropic regularization parameter
L: number of Sinkhorn iterations
MC: Separated Cache Dictionary

returns the Wasserstein barycenter μ as an nxn matrix

TODO: higher dimensions than d=2, number of grid points different in dimensions/marginals
"""

function sinkhorn_barycenter_separated(  λ::AbstractVector{T}, α::AbstractVector{AT},
                                            b_₀, d₀,
                                            k, SP, caches,
                                            ) where {T, V, AT <: AbstractArray{V}}

    MC = caches.MC
    VMC = caches.VMC

    for s in 1:VMC.S
        VMC[:b,T][s] .= b_₀[s]
    end
    if SP.debias
        MC[:d,T] .= d₀
    end
    
    #=
    if SP.update_warmstart
        for s in 1:VMC.S
            VMC[:bL,T][s] .= VMC[:b,T][s]
        end
        if SP.debias
            MC[:dL,T] .= MC[:d,T]
        end
        SP.update_warmstart = false
    end

    if SP.warmstart == "ones"
        for s in 1:VMC.S
            VMC[:b,T][s] .= 1
        end
        if SP.debias
            MC[:d,T] .= 1
        end
    elseif SP.warmstart == "last_update"
        for s in 1:VMC.S
            VMC[:b,T][s] .= VMC[:bL,T][s]
        end
        if SP.debias
            VMC[:d,T] .= VMC[:dL,T] 
        end
    elseif SP.warmstart == "latest"
        nothing
    else
        error("No valid warmstart")
    end
    =#

    for l in 1:SP.L # Sinkhorn loop
       for s in 1:VMC.S
            apply_K_sep!(MC[:t2,T], VMC[:b,T][s], k, MC[:t1,T])
            MC[:t2,T] .= α[s] ./ MC[:t2,T]      # this is aˡₛ
            apply_K_sep!(VMC[:φ,T][s], MC[:t2,T], k, MC[:t1,T])
        end

        SP.debias ? MC[:μ,T] .= MC[:d,T] : MC[:μ,T] .= 1
        
        for s in 1:MC.S
            MC[:μ,T] .*= VMC[:φ,T][s] .^ λ[s]
        end
      
        for s in 1:MC.S
            VMC[:b,T][s] .= MC[:μ,T] ./ VMC[:φ,T][s]
        end

        if SP.debias # && ( l%2==0 || l==SP.L )
            apply_K_sep!(MC[:t2,T], MC[:d,T], k, MC[:t1,T])
            MC[:d,T] .= sqrt.( MC[:d,T] .* MC[:μ,T] ./ MC[:t2,T] )
        end
    end

    if SP.update_potentials
        for s in 1:VMC.S
            b_₀[s] .= ForwardDiff.value.(VMC[:b,T][s])
        end
        if SP.debias
            d₀ .= ForwardDiff.value.(MC[:d,T])
        end
    end

    return copy(MC[:μ,T])
end

function sinkhorn_barycenter_separated(  λ::AbstractVector{T}, α::AbstractVector{AT},
                                            k, SP, caches,
                                            ) where {T, V, AT <: AbstractArray{V}}
    sinkhorn_barycenter_separated(   λ, α,
                                        [ones(V, size(α[s])) for s in eachindex(α)], 
                                        ones(V, size(α[1])),
                                        k, SP, caches)
end
