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
sinkhorn_dvg_sep(p, β, a₀, b₀, d₁₀, d₂₀, k, SP, caches)
p: right marginal as nxn matrix
β: left marginal as nxn matrix
k: e^(c/ε) with cost c as dx(nxn) vector of matrices

Only differentiable by α

TODO: higher dimensions than d=2, number of grid points different in dimensions/marginals
"""

function sinkhorn_dvg_sep(α::AbstractArray{T}, β::AbstractArray{V},
                            a₀, b₀, dα₀, dβ₀,
                            k, SP, MC
                            ) where {T, V}

    k_min = deepcopy(k)

    ρ = 0.5

    ε_min = SP.ε
    ε_l = 1.0

    MC[:α,V] .= ForwardDiff.value.(α) 
    MC[:β,V] .= β

    # a = exp f/ε
    MC[:a,V] .= a₀
    MC[:b,V] .= b₀
    if SP.debias
        MC[:dα,V] .= dα₀
        MC[:dβ,V] .= dβ₀
    end

    for l in 1:SP.L

        if l != 1
            ε_l = maximum( (ε_min, ε_l * ρ) )
        end
        #if ε_l != ε_min
            for i in eachindex(k)
                k[i] .= k_min[i].^(ε_min/ε_l)
            end
        #end

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
        if l % 4 == 0 && ε_l == ε_min
            # updated b last, so check for marginal violation on a:
            MC[:βb,V] .= MC[:b,V] .* MC[:β,V] 
            apply_K_sep!(MC[:t2,V], MC[:βb,V], k, MC[:t1,V])
            MC[:t2,V] .*= MC[:α,V] .* MC[:a,V]
            MC[:t2,V] .-= MC[:α,V]

            if norm( MC[:t2,V] , 1) < SP.tol
                k .= k_min
                #println("break at $l")
                break # Sinkhorn loop
            end

            if l == SP.L
                k .= k_min
                #println("max iterations")
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
                        k, SP, MC
                        ) where {T, V}
    MC[:a,V] .= one(V)
    MC[:b,V] .= one(V)
    MC[:dα,V] .= one(V)
    MC[:dβ,V] .= one(V)
    sinkhorn_dvg_sep( α, β,
                        MC[:a,V], MC[:b,V],
                        MC[:dα,V], MC[:dβ,V],
                        k, SP, MC
                        )
end


function sinkhorn_barycenter_loop!(μ, λ::AbstractVector{T}, α, k, SP, MC) where {T}
    S = length(λ)

    # g[k] = SoftMin_α[k] [ C - f[k] ]
    # bₖ = 1 ⊘ K ★ (aₖ ⊙ αₖ)   
    @threads for s in 1:S
        MC[:αa,T] .= MC[:a,T,s] .* α[s]
        apply_K_sep!(MC[:t2,T], MC[:αa,T], k, MC[:t1,T])

        if SP.averaged_updates
            MC[:b₊,T,s] .= one(T) ./ MC[:t2,T]
        else
            MC[:b,T,s] .= one(T) ./ MC[:t2,T]
        end
    end

    # log μ = h / ε - ∑ₖ λ[k] g[k] / ε (h == 0 for no debiasing)
    # μ = d ⊘ ∏ₖ b[k] ^ λ[k]
    SP.debias ? μ .= MC[:d,T] : μ .= one(T)
    for s in 1:S
        μ .*= MC[:b,T,s] .^ (-λ[s])
    end

    # f[k] = SoftMin_μ [ C - g[k] ]
    # aₖ =  1 ⊘ K ★ (bₖ ⊙ μ)
    @threads for s in 1:S
        MC[:μb,T] .= MC[:b,T,s] .* μ
        apply_K_sep!(MC[:t2,T], MC[:μb,T], k, MC[:t1,T])    # this is 1 ⊘ a₊
        
        if SP.averaged_updates
            MC[:a,T,s] .= sqrt.(MC[:a,T,s] ./ MC[:t2,T])
            MC[:b,T,s] .= sqrt.(MC[:b,T,s] .* MC[:b₊,T,s])
        else
            MC[:a,T,s] .= one(T) ./ MC[:t2,T]
        end
    end

    # Debiasing
    if SP.debias

        apply_K_sep!(MC[:t2,T], MC[:d,T], k, MC[:t1,T])
        MC[:d,T] .= sqrt.( MC[:d,T] .* μ ./ MC[:t2,T] )

    end
end

function sinkhorn_barycenter_sep(λ::AbstractVector{T}, α::AbstractVector{AT},
                                    a_₀, d₀,
                                    k, SP, MC
                                    ) where {T, V, AT <: AbstractArray{V}}

    S = length(λ)
    μ = zeros(T, MC.n, MC.n)

    k_min = deepcopy(k)

    ρ = 0.5

    ε_min = SP.ε
    ε_l = 1.0

    # d = exp h / ε

    for s in 1:S
        MC[:a,T,s] .= a_₀[s]
        MC[:b,T,s] .= one(T)
    end
    if SP.debias
        MC[:d,T] .= d₀
    end

    # Sinkhorn loop
    for l in 1:SP.L

        if l != 1
            ε_l = maximum( (ε_min, ε_l * ρ) )
        end
        #if ε_l != ε_min
            for i in eachindex(k)
                k[i] .= k_min[i].^(ε_min/ε_l)
            end
        #end

        sinkhorn_barycenter_loop!(μ, λ, α, k, SP, MC)

        # Check for convergence every 4 iterations
        if l % 4 == 0 && ε_l == ε_min
            sum_norm = zero(V)
            # updated aₖ last, so check for marginal violation on bₖ:
            for s in 1:S
                MC[:αa,T] .= MC[:a,T,s] .* α[s]
                apply_K_sep!(MC[:t2,T], MC[:αa,T], k, MC[:t1,T])
                MC[:t2,T] .*= MC[:b,T,s] .* μ
                MC[:t2,T] .-= μ

                MC[:t2,V] .= ForwardDiff.value.(MC[:t2,T])
                sum_norm += norm(MC[:t2,V], 1)
                end

                # println("norm in iteration $l: $(sum_norm/S)")

                if sum_norm < SP.tol * S
                    #println("break at $l")
                    k .= k_min
                    break # Sinkhorn loop
                end

                if l == SP.L
                    k .= k_min
                    #println("max iterations")
                end
            end
        end

    if SP.update_potentials

        for s in 1:S
            a_₀[s] .= ForwardDiff.value.(MC[:a,T,s])
        end

        if SP.debias
            d₀ .= ForwardDiff.value.(MC[:d,T])
        end
    end

    return μ
end

function sinkhorn_barycenter_sep(  λ::AbstractVector{T}, α::AbstractVector{AT},
                                            k, SP, MC
                                            ) where {T, V, AT <: AbstractArray{V}}
    for s in eachindex(α)
        MC[:a,T,s] .= one(T)
    end
    MC[:d,T] .= one(T) 
    sinkhorn_barycenter_sep( λ, α,
                            [MC[:a,T,s] for s in eachindex(α)], 
                            MC[:d,T],
                            k, SP, MC)
end
