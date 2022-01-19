"""
get_cost_matrix_separated(n, d; a=0, b=0)
n: number of discretization points along one dimension
a: left interval bound
b: right interval bound

returns cost matrix C in separated form as a dx(nxn) vector of matrices

TODO: non-cube-domains, different n along every dimension
"""
function get_cost_matrix_separated(n, d; T=Float64, a=0, b=1)
    x₁ = range(a,b,length=n)
    c = [zeros(T,n,n) for _ in 1:d]
    for k in 1:d
        for i in 1:n
            for j in 1:n
                c[k][i,j] = sqeuclidean(x₁[i],x₁[j])
            end
        end
    end
    return c
end

function get_cost_matrix_separated_periodic(n, d)
    x₁ = range(0,1,length=n)
    c = [zeros(n,n) for _ in 1:d]
    for k in 1:d
        for i in 1:n
            for j in 1:n
                c[k][i,j] = minimum([sqeuclidean(x₁[i],x₁[j]), sqeuclidean(x₁[i]+1,x₁[j]), 
                                    sqeuclidean(x₁[i],x₁[j]+1), sqeuclidean(x₁[i]-1,x₁[j]),
                                    sqeuclidean(x₁[i],x₁[j]-1)])
            end
        end
    end
    return c
end

function get_cost_matrix_separated_halfperiodic(n,d)
    x₁ = range(0,1,length=n)
    c = [zeros(n,n) for _ in 1:2]
    for k in 1:2
        for i in 1:n
            for j in 1:n
                if k == 1
                    c[k][i,j] = minimum([sqeuclidean(x₁[i],x₁[j]), sqeuclidean(x₁[i]+1,x₁[j]), 
                                        sqeuclidean(x₁[i]-1,x₁[j])] )
                else
                    c[k][i,j] = sqeuclidean(x₁[i],x₁[j])
                end
            end
        end
    end
    return c
end

function get_gibbs_matrix(c::Vector{<: AbstractMatrix{T}}, ε) where T
    n = size(c[1],1)
    k = [zeros(T,n,n) for _ in 1:length(c)]
    for i in eachindex(c)
        k[i] .= get_gibbs_matrix(c[i], ε)
    end
    return k
end

"""
get_cost_matrix(c)

c: separated cost matrix as a dx(nxn) vector of matrices

returns cost matrix C in non-separated form as a NxN matrix

TODO: different n along every dimension, d > 2
"""
function get_cost_matrix(c)
    d = length(c)
    n, n = size(c[1])

    if d == 1
        return c[1]
    elseif d > 2
        error("unimplemented")
    else

        C = zeros(n^2,n^2)

        for i1 in 1:n
            for i2 in 1:n
                i = (i1-1)*n + i2
                for j1 in 1:n
                    for j2 in 1:n
                        j = (j1-1)*n + j2
                        C[i,j] = c[1][i1,j1] + c[2][i2,j2]
                    end
                end
            end
        end
        return C
    end
end

function get_gibbs_matrix(C::AbstractMatrix, ε)
    return exp.(-C./ε)
end


struct LazyCost{T,V,FT} <: AbstractMatrix{T}
    x::Matrix{T}
    y::Matrix{V}
    c::FT

    function LazyCost(x::Matrix{T}, y::Matrix{V}, c::FT = (x,y) -> sqeuclidean(x,y) ) where {T, V, FT <: Base.Callable}
        new{T,V,FT}(x,y,c)
    end
end
  
Base.getindex(C::LazyCost{T}, i, j) where T = @views C.c(C.x[i,:], C.y[j,:])::T
Base.size(C::LazyCost) = (size(C.x,1), size(C.y,1))
