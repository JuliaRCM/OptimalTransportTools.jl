""" 
This maps (v, C, log_α) ↦ u = log.(α) - log.(Kb) = -log.(Kb./α) = log.(α./Kb)
"""
function sinkhorn_step_log!(u::AbstractArray{T}, v, C, εinv, log_α, temp) where T
    for i in eachindex(u)
        for j in eachindex(v)          # sum over j
            temp[j] = C[i,j]
        end
        rmul!(temp, -εinv)
        temp .+= v .- log_α[i]
        u[i] = -logsumexp(temp)
    end
    return u
end

"""
wasserstein_distance_log(x, y, log_α, log_β, c, u₀, v₀, log_d₁₀, log_d₂₀, SP, caches)

Differentiable by x only!
"""
function sinkhorn_dvg_particles(    x::AbstractMatrix{T},
                                    y::AbstractMatrix{V},
                                    log_α::Vector{V},
                                    log_β::Vector{V},
                                    c::Base.Callable,
                                    u₀, v₀,
                                    log_d₁₀, log_d₂₀,
                                    SP, caches   ) where {T,V}

    VC = caches.VC
    εinv = 1/SP.ε

    C_xy = ForwardDiff.value.( LazyCost(x, y, c) )
    C_xx = ForwardDiff.value.( LazyCost(x, x, c) )
    C_yy = LazyCost(y, y, c)

    VC[:u,V] .= u₀
    VC[:v,V] .= v₀
    if SP.debias
        VC[:log_d₁,V] .= log_d₁₀
        VC[:log_d₂,V] .= log_d₂₀
    end
    
    for l in 1:SP.L
        sinkhorn_step_log!(VC[:u,V], VC[:v,V], C_xy, εinv, log_α, VC[:t1,V]) # u = log.(α./Kb)
        sinkhorn_step_log!(VC[:v,V], VC[:u,V], C_xy', εinv, log_β, VC[:t1,V]) # v = log.(β./Ka) assuming K = Kᵀ

        if SP.debias # && (l%2==0 || l==SP.L)
            VC[:t3,V] .= VC[:log_d₁,V]
            VC[:t2,V] .= VC[:log_d₁,V] .+ log_α
            sinkhorn_step_log!(VC[:log_d₁,V], VC[:t3,V], C_xx, εinv, VC[:t2,V], VC[:t1,V]) .*= 0.5

            VC[:t3,V] .= VC[:log_d₂,V]
            VC[:t2,V] .= VC[:log_d₂,V] .+ log_β
            sinkhorn_step_log!(VC[:log_d₂,V], VC[:t3,V], C_yy, εinv, VC[:t2,V], VC[:t1,V]) .*= 0.5
        end
    end

    C_xy = LazyCost(x, y, c)
    C_xx = LazyCost(x, x, c)

    S_ε = 0

    # calculate ε⟨α,u-d₁⟩ + ε⟨β,v-d₂⟩ - ε ∑ e^(u⊕v-C/ε) + 0.5ε ∑ e^(d₁⊕d₁-C/ε) + 0.5ε ∑ e^(d₂⊕d₂-C/ε)

    for i in eachindex(log_α)
        S_ε += SP.ε * exp(log_α[i]) * VC[:u,V][i]
        if SP.debias
            S_ε -= SP.ε * exp(log_α[i]) * VC[:log_d₁,V][i]
            for i_ in eachindex(log_α)
                S_ε += 0.5*SP.ε * exp( VC[:log_d₁,V][i] + VC[:log_d₁,V][i_] - C_xx[i,i_]*εinv )           
            end
        end
        for j in eachindex(log_β)
            S_ε -= SP.ε * exp( VC[:u,V][i] + VC[:v,V][j] - C_xy[i,j] * εinv )  
        end
    end

    for j in eachindex(log_β)
        S_ε += SP.ε * exp(log_β[j]) * VC[:v,V][j]
        if SP.debias
            S_ε -= SP.ε * exp(log_β[j]) * VC[:log_d₂,V][j]
            for j_ in eachindex(log_β)
                S_ε += 0.5*SP.ε * exp( VC[:log_d₂,V][j] + VC[:log_d₂,V][j_] - C_yy[j,j_]*εinv )           
            end
        end
    end

    return S_ε
end