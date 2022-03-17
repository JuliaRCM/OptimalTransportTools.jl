"""
get_cost_matrix_separated(n, d; a=0, b=0)
n: number of discretization points along one dimension
a: left interval bound
b: right interval bound

returns cost matrix C in separated form as a dx(nxn) vector of matrices

TODO: non-cube-domains, different n along every dimension
"""
function get_cost_matrix_separated(n, d; a=zeros(d), b=ones(d))
    c = [zeros(n,n) for _ in 1:d]
    for k in 1:d
        x₁ = range(a[k],b[k],length=n)
        for i in 1:n
            for j in 1:n
                c[k][i,j] = sqeuclidean(x₁[i],x₁[j])
            end
        end
    end
    return c
end

function get_cost_matrix_separated_periodic(n, d; a=zeros(d), b=ones(d))
    c = [zeros(n,n) for _ in 1:d]
    for k in 1:d
        x₁ = range(a[k],b[k],length=n)
        for i in 1:n
            for j in 1:n
                c[k][i,j] = minimum([sqeuclidean(x₁[i],x₁[j]), sqeuclidean(x₁[i]+b[k]-a[k],x₁[j]), 
                                        sqeuclidean(x₁[i]-b[k]+a[k],x₁[j])] )
            end
        end
    end
    return c
end

function get_cost_matrix_separated_halfperiodic(n, d; a=zeros(d), b=ones(d))
    c = [zeros(n,n) for _ in 1:2]
    for k in 1:2
        x₁ = range(a[k],b[k],length=n)
        for i in 1:n
            for j in 1:n
                if k == 1
                    c[k][i,j] = minimum([sqeuclidean(x₁[i],x₁[j]), sqeuclidean(x₁[i]+b[k]-a[k],x₁[j]), 
                                        sqeuclidean(x₁[i]-b[k]+a[k],x₁[j])] )
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
  
Base.getindex(C::LazyCost{T,V}, i, j) where {T,V} = @views C.c(C.x[i,:], C.y[j,:])::promote_type(T,V)
Base.size(C::LazyCost) = (size(C.x,1), size(C.y,1))
