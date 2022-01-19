"""
applies a ↦ b = Ka
"""
@inline function  apply_K_sep!(b::AbstractMatrix, a, k, tmp)
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
wasserstein_distance_separated(p, β, k, ε, L)
p: right marginal as nxn matrix
β: left marginal as nxn matrix
C: Cost as dx(nxn) vector of matrices
ε: entropic regularization parameter
L: number of Sinkhorn iterations

returns a tupel of entropic Wasserstein distance W and scalings a,b as nxn matrices

TODO: higher dimensions than d=2, number of grid points different in dimensions/marginals
"""

function sinkhorn_dvg_separated(α::AbstractArray{T}, β::AbstractArray{T₂},
                                        a₀, b₀, d₁₀, d₂₀,
                                        k, SP, caches
                                        ) where {T, T₂}

    MC = caches.MC

    MC[:a,T] .= a₀;   MC[:b,T] .= b₀
    if SP.debias
        MC[:d₁,T] .= d₁₀; MC[:d₂,T] .= d₂₀
    end
   
    for l in 1:SP.L
        if SP.averaged_updates
            apply_K_sep!(MC[:t2,T], MC[:b,T], k, MC[:t1,T])
            MC[:a₊,T] .= α ./ MC[:t2,T]
            apply_K_sep!(MC[:t2,T], MC[:a,T], k, MC[:t1,T])
            MC[:b₊,T] .= β ./ MC[:t2,T]
        
            MC[:b,T] .= sqrt.(MC[:b,T] .* MC[:b₊,T])
            MC[:a,T] .= sqrt.(MC[:a,T] .* MC[:a₊,T])            
        else
            apply_K_sep!(MC[:t2,T], MC[:b,T], k, MC[:t1,T])
            MC[:a,T] .= α ./ MC[:t2,T]
            apply_K_sep!(MC[:t2,T], MC[:a,T], k, MC[:t1,T])
            MC[:b,T] .= β ./ MC[:t2,T]
        end

        if SP.debias #&& ( l%2==0 || l==SP.L )
            if SP.averaged_updates
                apply_K_sep!(MC[:t2,T], MC[:d₁,T], k, MC[:t1,T])
                MC[:d₁₊,T] .= sqrt.( MC[:d₁,T] .* α ./ MC[:t2,T] )
                apply_K_sep!(MC[:t2,T], MC[:d₂,T], k, MC[:t1,T])
                MC[:d₂₊,T] .= sqrt.( MC[:d₂,T] .* β ./ MC[:t2,T] )

                MC[:d₁,T] .= sqrt.(MC[:d₁,T] .* MC[:d₁₊,T])
                MC[:d₂,T] .= sqrt.(MC[:d₂,T] .* MC[:d₂₊,T]) 
            else
                apply_K_sep!(MC[:t2,T], MC[:d₁,T], k, MC[:t1,T])
                MC[:d₁,T] .= sqrt.( MC[:d₁,T] .* α ./ MC[:t2,T] )
                apply_K_sep!(MC[:t2,T], MC[:d₂,T], k, MC[:t1,T])
                MC[:d₂,T] .= sqrt.( MC[:d₂,T] .* β ./ MC[:t2,T] )
            end
        end
    end

    if SP.debias
        for i in eachindex(MC[:t2,T])
                MC[:t2,T][i] =  α[i] * (_safe_log( ForwardDiff.value( MC[:a,T][i] )) 
                                        - _safe_log( ForwardDiff.value( MC[:d₁,T][i] ))) +
                                β[i] * (_safe_log( ForwardDiff.value( MC[:b,T][i] )) 
                                        - _safe_log( ForwardDiff.value( MC[:d₂,T][i] ))) 
        end
    else
        for i in eachindex(MC[:t2,T])
                MC[:t2,T][i] =  α[i] * _safe_log( ForwardDiff.value( MC[:a,T][i] )) +
                                β[i] * _safe_log( ForwardDiff.value( MC[:b,T][i] ))
        end
    end

    S_ε = SP.ε * sum( MC[:t2,T] )

    if SP.update_potentials
        a₀ .= ForwardDiff.value.(MC[:a,T]) 
        b₀ .= ForwardDiff.value.(MC[:b,T])
        if SP.debias
            d₁₀ .= ForwardDiff.value.(MC[:d₁,T]) 
            d₂₀ .= ForwardDiff.value.(MC[:d₂,T])
        end
    end
    
    return S_ε
end

function sinkhorn_dvg_separated(α::AbstractArray{T}, β::AbstractArray{T₂},
                                        k, SP, caches
                                        ) where {T, T₂}
    sinkhorn_dvg_separated( α, β,
                                    ones(T₂, size(α)), ones(T₂, size(β)),
                                    ones(T₂, size(α)), ones(T₂, size(β)),
                                    k, SP, caches)
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
                                            ) where {T, T₂, AT <: AbstractArray{T₂}}

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
                                            ) where {T, T₂, AT <: AbstractArray{T₂}}
    sinkhorn_barycenter_separated(   λ, α,
                                        [ones(T₂, size(α[s])) for s in eachindex(α)], 
                                        ones(T₂, size(α[1])),
                                        k, SP, caches)
end
