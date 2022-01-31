""" 
This maps (v, c, log_α) ↦ u = log.(p) - log.(Kb) = -log.(Kb./p) = log.(p./Kb)
"""
function sinkhorn_step_logsep!(u::AbstractArray{T}, v, c, εinv, log_α, tempv, tempm) where T
    for i₁ in eachindex(tempv), j₂ in eachindex(tempv)     # sum over j₁
        tempv .= @view c[1][i₁,:]
        rmul!(tempv, -εinv)
        tempv .+= @view v[:,j₂]     # using c = cᵀ
        tempm[i₁,j₂] = logsumexp(tempv)
    end
    for i₁ in eachindex(tempv), i₂ in eachindex(tempv)     # sum over j₂
        tempv .= @view c[2][i₂,:]
        rmul!(tempv, -εinv)
        tempv .-= log_α[i₁,i₂]
        tempv .+= @view tempm[i₁,:]
        u[i₁,i₂] = -logsumexp(tempv)
    end
    return u
end

"""
wasserstein_distance_logseparated(log_α, log_β, u₀, v₀, log_d₁₀, log_d₂₀, c, SP, caches)
log_α: right marginal weights as nxn matrix in log-scale
log_β: left marginal weights as nxn matrix in log-scale
c: Cost between grid points as dx(nxn) vector of matrices

Only differentiable by log_α

TODO: higher dimensions than d=2, number of grid points different in dimensions/marginals
"""

function sinkhorn_dvg_logseparated( log_α::AbstractArray{T}, log_β::AbstractArray{V}, 
                                    u₀, v₀, log_d₁₀, log_d₂₀,
                                    c, SP, caches,
                                    ) where {T, V}

    MC = caches.MC
    VC = caches.VC
    εinv = 1/SP.ε

    MC[:log_α,V] .= ForwardDiff.value.(log_α) # Value of input weights to speed up the iterations

    MC[:u,V] .= u₀;   MC[:v,V] .= v₀
    if SP.debias
        MC[:log_d₁,V] .= log_d₁₀; MC[:log_d₂,V] .= log_d₂₀
    end
    
    for l in 1:SP.L
        if SP.averaged_updates
            sinkhorn_step_logsep!(MC[:u₊,V], MC[:v,V], c, εinv, MC[:log_α,V], VC[:t1,V], MC[:t1,V]) # u = log.(p./Kb)
            sinkhorn_step_logsep!(MC[:v₊,V], MC[:u,V], c, εinv, log_β, VC[:t1,V], MC[:t1,V]) # v = log.(q./Ka) assuming K = Kᵀ
            
            MC[:u,V] .= 0.5 * (MC[:u₊,V] + MC[:u,V])
            MC[:v,V] .= 0.5 * (MC[:v₊,V] + MC[:v,V])
        else
            sinkhorn_step_logsep!(MC[:u,V], MC[:v,V], c, εinv, MC[:log_α,V], VC[:t1,V], MC[:t1,V]) # u = log.(p./Kb)
            sinkhorn_step_logsep!(MC[:v,V], MC[:u,V], c, εinv, log_β, VC[:t1,V], MC[:t1,V]) # v = log.(q./Ka) assuming K = Kᵀ
        end

        if SP.debias # && (l%2==0 || l==SP.L)
            if SP.averaged_updates
                MC[:t3,V] .= MC[:log_d₁,V]
                MC[:t2,V] .= MC[:log_d₁,V] .+ MC[:log_α,V] 
                sinkhorn_step_logsep!(MC[:log_d₁₊,V], MC[:t3,V], c, εinv, 
                                        MC[:t2,V], VC[:t1,V], MC[:t1,V]) .*= 0.5

                MC[:t3,V] .= MC[:log_d₂,V] 
                MC[:t2,V] .= MC[:log_d₂,V] .+ log_β
                sinkhorn_step_logsep!(MC[:log_d₂₊,V], MC[:t3,V], c, εinv, 
                                    MC[:t2,V], VC[:t1,V], MC[:t1,V]) .*= 0.5

                MC[:log_d₂,V] .= 0.5 * (MC[:log_d₂₊,V] + MC[:log_d₂,V])
                MC[:log_d₁,V] .= 0.5 * (MC[:log_d₁₊,V] + MC[:log_d₁,V])
            else
                MC[:t3,V] .= MC[:log_d₁,V]
                MC[:t2,V] .= MC[:log_d₁,V] .+ MC[:log_α,V] 
                sinkhorn_step_logsep!(MC[:log_d₁,V], MC[:t3,V], c, εinv, 
                                        MC[:t2,V], VC[:t1,V], MC[:t1,V]) .*= 0.5

                MC[:t3,V] .= MC[:log_d₂,V] 
                MC[:t2,V] .= MC[:log_d₂,V] .+ log_β
                sinkhorn_step_logsep!(MC[:log_d₂,V], MC[:t3,V], c, εinv, 
                                    MC[:t2,V], VC[:t1,V], MC[:t1,V]) .*= 0.5
            end
        end
    end

    if SP.debias
        for i in eachindex(MC[:t2,T])
            MC[:t2,T][i] =  exp(log_α[i]) * ( MC[:u,V][i] - MC[:log_d₁,V][i] ) +
                            exp(log_β[i]) * ( MC[:v,T][i] - MC[:log_d₂,T][i] )
        end
    else
        for i in eachindex(MC[:t2,T])
            MC[:t2,T][i] =  exp(log_α[i]) * MC[:u,V][i] + exp(log_β[i]) * MC[:v,V][i]
        end
    end
    S_ε = SP.ε * sum( MC[:t2,T] )

    if SP.update_potentials
        u₀ .= MC[:u,V]
        v₀ .= MC[:v,V]
        if SP.debias
            log_d₁₀ .= MC[:log_d₁,V] 
            log_d₂₀ .= MC[:log_d₂,V]
        end
    end

    return S_ε
end

function sinkhorn_dvg_logseparated( log_α::AbstractArray{T}, log_β::AbstractArray{V},     
                                            c, SP, caches,
                                            ) where {T, V}
            sinkhorn_dvg_logseparated(  log_α, log_β,
                                        zeros(V, size(log_α)), zeros(V, size(log_β)),
                                        zeros(V, size(log_α)), zeros(V, size(log_β)),
                                        c, SP, caches)
end

function sinkhorn_barycenter_logseparated(λ::AbstractVector{T}, log_α::AbstractVector{AT},
                                            v₀, log_d₀,
                                            c, SP, caches,
                                            ) where {T, V, AT <: AbstractArray{V}}

    MC = caches.MC
    VC = caches.VC
    VMC = caches.VMC
    εinv = 1/SP.ε

    for s in 1:VMC.S
        VMC[:v,T][s] .= v₀[s]
    end
    if SP.debias
        MC[:log_d,T] .= log_d₀
    end

    #=
    if SP.update_warmstart
        for s in 1:VMC.S
            VMC[:vL,T][s] .= VMC[:v,T][s]
        end
        if SP.debias
            MC[:log_dL,T] .= MC[:log_d,T]
        end
        SP.update_warmstart = false
    end

    if SP.warmstart == "ones"
        for s in 1:VMC.S
            VMC[:v,T][s] .= 0
        end
        if SP.debias
            MC[:log_d,T] .= 0
        end
    elseif SP.warmstart == "last_update"
        for s in 1:VMC.S
            VMC[:v,T][s] .= VMC[:vL,T][s]
        end
        if SP.debias
            VMC[:log_d,T] .= VMC[:log_dL,T] 
        end
    elseif SP.warmstart == "latest"
        nothing
    else
        error("No valid warmstart")
    end
    =#

    for l in 1:SP.L # Sinkhorn loop
        
        MC[:t3,T] .= 0

       for s in 1:VMC.S
            sinkhorn_step_logsep!(MC[:t2,T], VMC[:v,T][s], c, εinv, log_α[s], VC[:t1,T], MC[:t1,T])   # this is uˡₛ
            sinkhorn_step_logsep!(VMC[:log_φ,T][s], MC[:t2,T], c, εinv, MC[:t3,T], VC[:t1,T], MC[:t1,T]) .*= -1   # log φˡₛ = log Kᵀ aˡₛ
        end

        SP.debias ? MC[:log_μ,T] .= MC[:log_d,T] : MC[:log_μ,T] .= 0
        
        for s in 1:MC.S
            MC[:log_μ,T] .+= λ[s] .* VMC[:log_φ,T][s]
        end
      
        for s in 1:MC.S
            VMC[:v,T][s] .= MC[:log_μ,T] .- VMC[:log_φ,T][s]
        end

        if SP.debias #&& ( l%2==0 || l==SP.L )
            MC[:t3,T] .= MC[:log_d,T] 
            MC[:t2,T] .= MC[:log_d,T] .+ MC[:log_μ,T] 
            sinkhorn_step_logsep!(MC[:log_d,T], MC[:t3,T], c, εinv, MC[:t2,T], VC[:t1,T], MC[:t1,T]) .*= 0.5
        end
    end

    if SP.update_potentials
        for s in 1:VMC.S
            v₀[s] .= ForwardDiff.value.( VMC[:v,T][s] )
        end
        if SP.debias
            log_d₀ .= ForwardDiff.value.( MC[:log_d,T] )
        end
    end

    return copy(MC[:log_μ,T])
end

function sinkhorn_barycenter_logseparated(λ::AbstractVector{T}, log_α::AbstractVector{AT},
                                            c, SP, caches,
                                            ) where {T, V, AT <: AbstractArray{V}}
    sinkhorn_barycenter_logseparated(λ, log_α,
                                        [zeros(V, size(log_α[s])) for s in eachindex(log_α)],
                                        zeros(V, size(log_α[1])),
                                        c, SP, caches)
end