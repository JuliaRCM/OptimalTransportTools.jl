function rho_divergence(a,b)
    b .- a .+ a .* log.( a./b )
end

"""
wasserstein_distance_greenkhorn(p, q, C, ε, L)
p: right marginal N vector
q: left marginal N vector
C: Cost NxN matrix
ε: entropic regularization parameter
L: number of Sinkhorn iterations

returns a tupel of entropic Wasserstein distance W and scaling potentials u,v as N vectors

TODO: number of grid points different in marginals
"""
function wasserstein_distance_greenkhorn(p, q, C, ε, L)
    
    @assert length(p) == size(C,1)
    @assert length(q) == size(C,2)

    a = ones(length(p)) # a = exp(u)
    b = ones(length(q)) # b = exp(v)

    K = get_gibbs_matrix(C, ε)
    
    local r_Gamma = K * b # initial row sum
    local c_Gamma = K * a # initial column sum

    for l in 1:L
        local m_i, I = findmax( rho_divergence(p, r_Gamma) )
        local m_j, J = findmax( rho_divergence(q, c_Gamma) )
        if m_i > m_j
            a_old = a[I]
            a[I] *= p[I] / r_Gamma[I] # update Ith entry of left scaling vector
            r_Gamma[I] *= a[I] / a_old # update row sum
            # c_Gamma .+= (a[I] - a_old) .* K[I,:] .* b # update column sum - O(N)
            c_Gamma .= b .* K * a # O(N^2)
        else
            b_old = b[J]
            b[J] *= q[J] / c_Gamma[J] # update Jth entry of right scaling vector
            c_Gamma[J] *= b[J] / b_old # update column sum
            # r_Gamma .+= a .* K[:,J] .* (b[J] - b_old) # update row sum
            r_Gamma .= a .* K * b
        end
    end

    return get_distance_from_scaling(a, b, C, ε), a, b
end


"""
wasserstein_distance_greenkhorn_logstable(p, q, C, ε, L)
p: right marginal N vector
q: left marginal N vector
C: Cost NxN matrix
ε: entropic regularization parameter
L: number of Sinkhorn iterations

returns a tupel of entropic Wasserstein distance W and scaling potentials u,v as N vectors

TODO: number of grid points different in marginals
"""
function wasserstein_distance_greenkhorn_logstable(p, q, C, ε, L)
    
    @assert length(p) == size(C,1)
    @assert length(q) == size(C,2)

    u = zeros(length(p)) # a = exp(u)
    v = zeros(length(q)) # b = exp(v)
    log_p = log.(p)
    log_q = log.(q)

    K = get_gibbs_matrix(C, ε)
    
    local r_Gamma = K * exp.(v) # initial row sum
    local c_Gamma = K * exp.(u) # initial column sum

    for l in 1:L
        local m_i, I = findmax( rho_divergence(p, r_Gamma) )
        local m_j, J = findmax( rho_divergence(q, c_Gamma) )
        if m_i > m_j
            u_old = u[I]
            u[I] += log_p[I] - log(r_Gamma[I]) # update Ith entry of left scaling potential
            r_Gamma[I] *= exp(u[I] - u_old) # update row sum
            @. c_Gamma += exp(u[I] - C[I,:]/ε + v) - exp(u_old - C[I,:]/ε + v) # update column sum
        else
            v_old = v[J]
            v[J] += log_q[J] - log(c_Gamma[J]) # update Jth entry of right scaling vector
            c_Gamma[J] *= exp(v[J] - v_old) # update column sum
            @. r_Gamma += exp(u - C[:,J]/ε + v[J]) - exp(u - C[:,J]/ε + v_old) # update row sum
        end
    end

    return get_distance_from_scaling_logstable(u, v, C, ε), u, v
end
