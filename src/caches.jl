struct CacheDict{T <: Array, N}
    S::Int
    n::Int
    caches::Dict{UInt64, Array}
    
    function CacheDict{T,N}(S,n) where {T,N}
        new{T,N}(S, n, Dict{UInt64, Array}())
    end
end

const ArrayCache{N} = CacheDict{Array{T,N} where {T <: Number, N}}
const VectorCache = ArrayCache{1}
const MatrixCache = ArrayCache{2}

const VectorArrayCache{N} = CacheDict{Vector{Array{T,N} where T <: Number}, N}
const VectorMatrixCache = VectorArrayCache{2}
const VectorVectorCache = VectorArrayCache{1}

function _new_cache_entry(c::ArrayCache{N}, ST::DataType) where {N}
    ones(ST, Tuple([c.n for _ in 1:N])...)
end

function _new_cache_entry(c::VectorArrayCache{N}, ST::DataType) where {N}
    [ones(ST, Tuple([c.n for _ in 1:N])...) for _ in 1:c.S]
end

_get_type(::ArrayCache{N}, ST) where {N} = Array{ST,N}
_get_type(::VectorArrayCache{N}, ST) where {N} = Vector{Array{ST,N}}

@inline function Base.getindex(c::CacheDict, name::Symbol, ST::DataType)
    key = hash(Threads.threadid(), hash(name, hash(ST)))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = _new_cache_entry(c, ST)
    end::_get_type(c, ST)
end