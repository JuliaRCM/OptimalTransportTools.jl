struct CacheDict{T <: Array, N}
    n::Int
    caches::Dict{UInt64, Array}
    
    function CacheDict{T,N}(n) where {T,N}
        new{T,N}(n, Dict{UInt64, Array}())
    end
end

const ArrayCache{N} = CacheDict{Array{T,N} where {T <: Number, N}}
const VectorCache = ArrayCache{1}
const MatrixCache = ArrayCache{2}

@inline function Base.getindex(c::CacheDict{T,N}, name::Symbol, ST::DataType, i::Int=Threads.threadid()) where {T,N}
    key = hash(i, hash(name, hash(ST)))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = ones(ST, Tuple([c.n for _ in 1:N])...)
    end::Array{ST,N}
end
