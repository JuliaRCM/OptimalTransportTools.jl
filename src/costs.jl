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
                c[k][i,j] = 0.5 * sqeuclidean(x₁[i],x₁[j])
            end
        end
    end
    return c
end

function get_cost_matrix_separated(x, y, d)
    c = [zeros( length(x[k]),length(y[k])) for k in 1:d]
    for k in 1:d
        for i in eachindex(x[k])
            for j in eachindex(y[k])
                c[k][i,j] = 0.5 * sqeuclidean(x[k][i],y[k][j])
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
                c[k][i,j] = 0.5 * minimum([sqeuclidean(x₁[i],x₁[j]), sqeuclidean(x₁[i]+b[k]-a[k],x₁[j]),
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
                    c[k][i,j] = 0.5 * minimum([sqeuclidean(x₁[i],x₁[j]), sqeuclidean(x₁[i]+b[k]-a[k],x₁[j]), sqeuclidean(x₁[i]-b[k]+a[k],x₁[j])] )
                else
                    c[k][i,j] = 0.5 * sqeuclidean(x₁[i],x₁[j])
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


struct LazyCost{T,N,XT,YT,FT} <: AbstractMatrix{T}
    x::Array{XT,N} # N × d
    y::Array{YT,N}
    c::FT

    function LazyCost(x::Array{XT,N}, y::Array{YT,N}, c::FT = (x,y) -> 0.5 * sqeuclidean(x,y) ) where {XT, YT, N, FT <: Base.Callable}
        t = c(x[begin], y[begin])
        new{typeof(t),N,XT,YT,FT}(x,y,c)
    end
end

Base.getindex(C::LazyCost{T,1}, i, j) where {T} = C.c(C.x[i], C.y[j])::T
Base.getindex(C::LazyCost{T,N}, i, j) where {T,N} = @views C.c(C.x[i,:], C.y[j,:])::T
Base.size(C::LazyCost) = (size(C.x,1), size(C.y,1))
