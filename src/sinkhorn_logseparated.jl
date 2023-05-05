
""" 
This maps (f, c, log_α) ↦ g = - ε log.(∑ₓ α .* exp ε⁻¹ ( f - C(.,y) ))
"""
function softmin_separated!(f::AbstractArray{T}, g, log_α, ε, c, tempv, tempm) where T
    for i₁ in axes(tempm,1), j₂ in axes(tempm,2)  # sum over j₁
        @views tempv[1,:] .= -1/ε .* c[1][i₁,:]
        @views tempv[1,:] .+= 1/ε .* g[:,j₂]     # using c = cᵀ
        @views tempv[1,:] .+= log_α[:,j₂]
        @views tempm[i₁,j₂] = logsumexp(tempv[1,:])
    end
    for i₁ in axes(f,1), i₂ in axes(f,2)   # sum over j₂
        @views tempv[1,:] .= -1/ε .* c[2][i₂,:]
        @views tempv[1,:] .+= tempm[i₁,:]
        @views f[i₁,i₂] = - ε * logsumexp(tempv[1,:])
    end
end

"""
sinkhorn_dvg_logsep(log_α, log_β, u₀, v₀, log_d₁₀, log_d₂₀, c, SP, caches)
log_α: right marginal weights as nxn matrix in log-scale
log_β: left marginal weights as nxn matrix in log-scale
c: Cost between grid points as dx(nxn) vector of matrices

Only differentiable by log_α

TODO:   higher dimensions than d=2, number of grid points different in dimensions/marginals
        this method assumes Cxx == Cxy == Cyx and also all C symmetric
"""

function sinkhorn_dvg_logsep( log_α::AbstractArray{T}, log_β::AbstractArray{V}, 
                                f₀, g₀, hα₀, hβ₀,
                                c, SP, MC,
                                ) where {T, V}
    ε = SP.ε

    # Use the value of input weights to speed up the iterations - envelope theorem
    MC[:log_α,V] .= ForwardDiff.value.(log_α) 
    MC[:log_β,V] .= log_β

    # for convergence tests
    MC[:α,V] .= exp.(log_α) 
    MC[:β,V] .= exp.(log_β)

    MC[:f,V] .= f₀
    MC[:g,V] .= g₀
    if SP.debias
        MC[:hα,V] .= hα₀
        MC[:hβ,V] .= hβ₀
    end
    
    for l in 1:SP.L

        # for relative convergence check
        MC[:f₋,V] .= MC[:f,V]

        if SP.averaged_updates
            # f = 0.5 * f + 0.5 * SoftMin_β [ Cyx - g ]
            softmin_separated!(MC[:f₊,V], MC[:g,V] , MC[:log_β,V], ε, c, MC[:t1,V], MC[:t2,V])
            # g = 0.5 * g + 0.5 * SoftMin_α [ Cxy - f ]
            softmin_separated!(MC[:g₊,V], MC[:f,V] , MC[:log_α,V], ε, c, MC[:t1,V], MC[:t2,V])
            MC[:f,V] .= 0.5 .* (MC[:f₊,V] .+ MC[:f,V])
            MC[:g,V] .= 0.5 .* (MC[:g₊,V] .+ MC[:g,V])
        else
            softmin_separated!(MC[:f,V], MC[:g,V] , MC[:log_β,V], ε, c, MC[:t1,V], MC[:t2,V])
            softmin_separated!(MC[:g,V], MC[:f,V] , MC[:log_α,V], ε, c, MC[:t1,V], MC[:t2,V])
        end

        # Debiasing
        if SP.debias

            # hα = 0.5 * hα + 0.5 * SoftMin_α [ Cxx - hα ]
            softmin_separated!(MC[:hα₊,V], MC[:hα,V], MC[:log_α,V], ε, c, MC[:t1,V], MC[:t2,V])
            MC[:hα,V] .= 0.5 .* (MC[:hα₊,V] .+ MC[:hα,V])

            # hβ = 0.5 * hβ + 0.5 * SoftMin_β [ Cyy - hβ ]
            softmin_separated!(MC[:hβ₊,V], MC[:hβ,V], MC[:log_β,V], ε, c, MC[:t1,V], MC[:t2,V])
            MC[:hβ,V] .= 0.5 .* (MC[:hβ₊,V] .+ MC[:hβ,V])

        end

        # Check for convergence every _ iterations
        if l % 4 == 0
            # updated b = exp g/ε last, so check for marginal violation on a = exp f/ε:
            softmin_separated!(MC[:f₊,V], MC[:g,V] , MC[:log_β,V], ε, c, MC[:t1,V], MC[:t2,V])
            @. MC[:t1,V] = MC[:α,V] * ( exp( (MC[:f,V] - MC[:f₊,V]) / ε) - one(V) )  # this is π1 - α

            # check relative difference
            #@. MC[:t1,V] = (MC[:f₋,V] - MC[:f,V]) * MC[:α,V]
            #@. MC[:t2,V] = MC[:f₋,V] * MC[:α,V]    # normalize

            if norm( MC[:t1,V], 1) < SP.tol
                break # Sinkhorn loop
            end
        end


    end # end Sinkhorn loop

    # calculate loss
    
    if SP.debias
        for i in eachindex(MC[:t1,T])
            MC[:t1,T][i] = exp(log_α[i]) * ( MC[:f,V][i] - MC[:hα,V][i] ) + exp(log_β[i]) * ( MC[:g,V][i] - MC[:hβ,V][i] )
        end
    else
        for i in eachindex(MC[:t1,T])
            MC[:t1,T][i] = exp(log_α[i]) * MC[:f,V][i] + exp(log_β[i]) * MC[:g,V][i]
        end
    end

    S_ε = sum( MC[:t1,T] )

    if SP.update_potentials
        f₀ .= MC[:f,V]
        g₀ .= MC[:g,V]
        if SP.debias
            hα₀ .= MC[:hα,V] 
            hβ₀ .= MC[:hβ,V]
        end
    end

    return S_ε
end

"""
Convenience function for no given initial potentials
"""
function sinkhorn_dvg_logsep( log_α::AbstractArray{T}, log_β::AbstractArray{V},     
                                    c, SP, MC,
                                    ) where {T, V}
    MC[:f,V] .= zero(V)
    MC[:g,V] .= zero(V)
    MC[:hα,V] .= zero(V)
    MC[:hβ,V] .= zero(V)
    sinkhorn_dvg_logsep(    log_α, log_β,
                            MC[:f,V], MC[:g,V],
                            MC[:hα,V], MC[:hβ,V],
                            c, SP, MC)
end

"""
!!! Averaged updates not implemented here
"""
function sinkhorn_barycenter_logsep(λ::AbstractVector{T}, log_α::AbstractVector{AT},
                                    f₀, h₀,
                                    c, SP, MC,
                                    ) where {T, V, AT <: AbstractArray{V}}

    S = length(λ)
    ε = SP.ε
    log_μ = zeros(T, MC.n, MC.n)

    for s in 1:S
        MC[:f,T,s] .= f₀[s]
    end
    if SP.debias
        MC[:h,T] .= h₀
    end
    MC[:zero,T] .= zero(T)

    for l in 1:SP.L # Sinkhorn loop

        # for convergence check
        for s in 1:S
            MC[:g₋,T,s] .= MC[:g,T,s]
        end

        # g[k] = SoftMin_α[k] [ C - f[k] ]
        @threads for s in 1:S
            if SP.averaged_updates
                softmin_separated!(MC[:g₊,T,s], MC[:f,T,s], log_α[s], ε, c, MC[:t1,T], MC[:t2,T])
            else
                softmin_separated!(MC[:g,T,s], MC[:f,T,s], log_α[s], ε, c, MC[:t1,T], MC[:t2,T])
            end
        end

        # log μ = h / ε - ∑ₖ λ[k] g[k] / ε (h == 0 for no debiasing)
        SP.debias ? log_μ .= MC[:h,T] ./ ε : log_μ .= 0
        for s in 1:S
            log_μ .-= λ[s] .* MC[:g,T,s] ./ ε
        end

        # f[k] = SoftMin_μ [ C - g[k] ]
        @threads for s in 1:S
            if SP.averaged_updates
                softmin_separated!(MC[:f₊,T,s], MC[:g,T,s], log_μ, ε, c, MC[:t1,T], MC[:t2,T])
                MC[:f,T,s] .= 0.5 .* (MC[:f₊,T,s] .+ MC[:f,T,s])
                MC[:g,T,s] .= 0.5 .* (MC[:g₊,T,s] .+ MC[:g,T,s])
            else
                softmin_separated!(MC[:f,T,s], MC[:g,T,s], log_μ, ε, c, MC[:t1,T], MC[:t2,T])
            end
        end

        # Debiasing
        if SP.debias
            # h = 0.5 * h + 0.5 * ε * log(μ) + 0.5 * SoftMin_μ [ Cxx - h ]
            softmin_separated!(MC[:h₊,T], MC[:h,T], MC[:zero,T], ε, c, MC[:t1,T], MC[:t2,T])
            MC[:h,T] .= 0.5 .* ( MC[:h₊,T] .+ MC[:h,T] .+ ε .* log_μ)
        end

        # Check for convergence every 4 iterations
        if l % 4 == 0
            sum_norm = zero(V)
            # updated aₖ last, so check for marginal violation on bₖ:
            for s in 1:S
                softmin_separated!(MC[:g₊,T], MC[:f,T,s], log_α[s], ε, c, MC[:t1,T], MC[:t2,T])
                MC[:t1,T] .= exp.( log_μ ) .* ( exp.( (MC[:g,T,s] .- MC[:g₊,T]) ./ ε) .- one(T) )  # this is π'1 - μ

                MC[:t1,V] .= ForwardDiff.value.(MC[:t1,T])
                sum_norm += norm(MC[:t1,V], 1)
            end

            # check for relative difference
            #for s in 1:S
            #    @. MC[:t1,T] = (MC[:g,T,s] - MC[:g₋,T,s]) * exp( log_μ )
            #    @. MC[:t2,T] = MC[:g₋,T,s] * exp( log_μ )   # normalize
            #    sum_norm += norm( ForwardDiff.value.(MC[:t1,T]), 1) / norm( ForwardDiff.value.(MC[:t2,T]), 1)
            #end

            # println("norm in iteration $l: $(sum_norm/S)")

            if sum_norm < SP.tol * S
                break # Sinkhorn loop
            end
        end

    end # Sinkhorn loop

        if SP.update_potentials
            for s in 1:S
                f₀[s] .= ForwardDiff.value.( MC[:f,T,s] )
            end
            if SP.debias
                h₀ .= ForwardDiff.value.( MC[:h,T] )
            end
        end

    return log_μ
end


function sinkhorn_barycenter_logsep(λ::AbstractVector{T}, log_α::AbstractVector{AT},
                                        c, SP, MC,
                                        ) where {T, V, AT <: AbstractArray{V}}
    for s in eachindex(log_α)
        MC[:f,T,s] .= zero(T)
    end
    MC[:h,T] .= zero(T) 
    sinkhorn_barycenter_logsep(λ, log_α,
                                [MC[:f,T,s] for s in eachindex(log_α)],
                                MC[:h,T],
                                c, SP, MC)
end
