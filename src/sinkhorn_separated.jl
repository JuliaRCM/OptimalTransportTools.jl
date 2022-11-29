"""
applies a ↦ b = Ka
"""
@inline function apply_K_sep!(b::AbstractMatrix, a, k, tmp)
    mul!(tmp, k[1], a)       # :t[1] = k[1] * :b
    mul!(b, tmp, k[2]')
end

"""
applies a ↦ b = Ka
"""
@inline function apply_K_sep!(b::AbstractVector, a, K, tmp)
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

function sinkhorn_dvg_sep(α::AbstractArray{T}, β::AbstractArray{V},
                            a₀, b₀, dα₀, dβ₀,
                            k, SP, caches
                            ) where {T, V}

    MC = caches.MC

    MC[:α,V] .= ForwardDiff.value.(α) 
    MC[:β,V] .= β

    # a = exp f/ε
    MC[:a,V] .= a₀;   MC[:b,V] .= b₀
    if SP.debias
        MC[:dα,V] .= dα₀; MC[:dβ,V] .= dβ₀
    end
   
    for l in 1:SP.L

        if SP.averaged_updates
            # f = 0.5 * f + 0.5 * SoftMin_β [ Cyx - g ]
            MC[:βb,V] .= MC[:b,V] .* MC[:β,V] 
            apply_K_sep!(MC[:t2,V], MC[:βb,V], k, MC[:t1,V])

            # g = 0.5 * g + 0.5 * SoftMin_α [ Cxy - f ]
            MC[:αa,V] .= MC[:a,V] .* MC[:α,V]
            apply_K_sep!(MC[:t3,V], MC[:αa,V], k, MC[:t1,V])
            
            MC[:a,V] .= sqrt.(MC[:a,V] ./ MC[:t2,V])
            MC[:b,V] .= sqrt.(MC[:b,V] ./ MC[:t3,V])

        else
            MC[:βb,V] .= MC[:b,V] .* MC[:β,V] 
            apply_K_sep!(MC[:t2,V], MC[:βb,V], k, MC[:t1,V])
            MC[:a,V] .= one(V) ./ MC[:t2,V]

            MC[:αa,V] .= MC[:a,V] .* MC[:α,V]
            apply_K_sep!(MC[:t2,V], MC[:αa,V], k, MC[:t1,V])
            MC[:b,V] .= one(V) ./ MC[:t2,V] 
        end

        if SP.debias
            MC[:βdβ,V] .= MC[:dβ,V] .* MC[:β,V]
            apply_K_sep!(MC[:t2,V], MC[:βdβ,V], k, MC[:t1,V])
            MC[:dβ,V] .= sqrt.( MC[:dβ,V] ./ MC[:t2,V] )

            MC[:αdα,V] .= MC[:dα,V] .* MC[:α,V]
            apply_K_sep!(MC[:t2,V], MC[:αdα,V], k, MC[:t1,V])
            MC[:dα,V] .= sqrt.( MC[:dα,V] ./ MC[:t2,V] )
        end

        # Check for convergence every 4 iterations
        if l % 4 == 0
            # updated b last, so check for marginal violation on a:
            MC[:βb,V] .= MC[:b,V] .* MC[:β,V] 
            apply_K_sep!(MC[:t2,V], MC[:βb,V], k, MC[:t1,V])
            MC[:t2,V] .*= MC[:α,V] .* MC[:a,V]
            MC[:t2,V] .-= MC[:α,V]

            # println("norm in iteration $l: $(norm( MC[:t2,V] , 1))")

            if norm( MC[:t2,V] , 1) < SP.tol
                break # Sinkhorn loop
            end
        end

    end # end Sinkhorn loop

    if SP.debias
        for i in eachindex(MC[:t2,T])
                MC[:t2,T][i] = ( α[i] * (log( MC[:a,V][i] ) - log( MC[:dα,V][i] )) 
                                + β[i] * (log( MC[:b,V][i] ) - log( MC[:dβ,V][i] )) )
        end
    else
        for i in eachindex(MC[:t2,T])
                MC[:t2,T][i] = α[i] * log( MC[:a,V][i] ) + β[i] * log( MC[:b,V][i] )
        end
    end

    S_ε = SP.ε * sum( MC[:t2,T] )

    if SP.update_potentials
        a₀ .= MC[:a,V]
        b₀ .= MC[:b,V]
        if SP.debias
            dα₀ .= MC[:dα,V]
            dβ₀ .= MC[:dβ,V]
        end
    end
    
    return S_ε
end

function sinkhorn_dvg_sep(α::AbstractArray{T}, β::AbstractArray{V},
                        k, SP, caches
                        ) where {T, V}
    sinkhorn_dvg_sep( α, β,
                            ones(V, size(α)), ones(V, size(β)),
                            ones(V, size(α)), ones(V, size(β)),
                            k, SP, caches
                            )
end


function sinkhorn_barycenter_sep(λ::AbstractVector{T}, α::AbstractVector{AT},
                                a_₀, d₀,
                                k, SP, caches,
                                ) where {T, V, AT <: AbstractArray{V}}

    MC = caches.MC
    VMC = caches.VMC

    # d = exp h / ε

    for s in 1:VMC.S
        VMC[:a,T][s] .= a_₀[s]
    end
    if SP.debias
        MC[:d,T] .= d₀
    end

    for l in 1:SP.L # Sinkhorn loop

    
        # g[k] = SoftMin_α[k] [ C - f[k] ]
        # bₖ = 1 ⊘ K ★ (aₖ ⊙ αₖ)   
        for s in 1:VMC.S
            MC[:αa,T] .= VMC[:a,T][s] .* α[s]
            apply_K_sep!(MC[:t2,T], MC[:αa,T], k, MC[:t1,T])

            if SP.averaged_updates
                VMC[:invb₊,T][s] .= MC[:t2,T]
            else
                VMC[:b,T][s] .= one(T) ./ MC[:t2,T]
            end

        end

        # log μ = h / ε - ∑ₖ λ[k] g[k] / ε (h == 0 for no debiasing)
        # μ = d ⊘ ∏ₖ b[k] ^ λ[k]
        SP.debias ? MC[:μ,T] .= MC[:d,T] : MC[:μ,T] .= one(T)
        for s in 1:MC.S
            MC[:μ,T] .*= VMC[:b,T][s] .^ (-λ[s])
        end

        # f[k] = SoftMin_μ [ C - g[k] ]
        # aₖ =  1 ⊘ K ★ (bₖ ⊙ μ)
        for s in 1:MC.S
            MC[:μb,T] .= VMC[:b,T][s] .* MC[:μ,T]
            apply_K_sep!(MC[:t2,T], MC[:μb,T], k, MC[:t1,T])    # this is 1 ⊘ a₊
            
            if SP.averaged_updates
                VMC[:a,T][s] .= sqrt.(VMC[:a,T][s] ./ MC[:t2,V])
                VMC[:b,T][s] .= sqrt.(VMC[:b,T][s] ./ VMC[:invb₊,T][s])
            else
                VMC[:a,T][s] .= one(T) ./ MC[:t2,T]
            end
        end

        # Debiasing
        if SP.debias

            apply_K_sep!(MC[:t2,T], MC[:d,T], k, MC[:t1,T])
            MC[:d,T] .= sqrt.( MC[:d,T] .* MC[:μ,T] ./ MC[:t2,T] )

        end

        # Check for convergence every 4 iterations
        if l % 4 == 0
            sum_norm = zero(V)
            # updated aₖ last, so check for marginal violation on bₖ:
            for s in 1:VMC.S
                MC[:αa,T] .= VMC[:a,T][s] .* α[s]
                apply_K_sep!(MC[:t2,T], MC[:αa,T], k, MC[:t1,T])
                MC[:t2,T] .*= VMC[:b,T][s] .* MC[:μ,T]
                MC[:t2,T] .-= MC[:μ,T]

                sum_norm += norm( ForwardDiff.value.(MC[:t2,T]) , 1)
            end

            # println("norm in iteration $l: $(sum_norm/MC.S)")

            if sum_norm < SP.tol * MC.S
                break # Sinkhorn loop
            end
        end

    end # end Sinkhorn loop

    if SP.update_potentials

        for s in 1:VMC.S
            a_₀[s] .= ForwardDiff.value.(VMC[:a,T][s])
        end

        if SP.debias
            d₀ .= ForwardDiff.value.(MC[:d,T])
        end
    end

    return copy(MC[:μ,T])
end

function sinkhorn_barycenter_sep(  λ::AbstractVector{T}, α::AbstractVector{AT},
                                            k, SP, caches,
                                            ) where {T, V, AT <: AbstractArray{V}}
    sinkhorn_barycenter_sep( λ, α,
                            [ones(V, size(α[s])) for s in eachindex(α)], 
                            ones(V, size(α[1])),
                            k, SP, caches)
end



### Old code

#=

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

=#