module WassersteinDictionaries

using ForwardDiff
using LinearAlgebra
using Distances
import SimpleSolvers as SS
using LogExpFunctions

using Printf

include("caches.jl")
export CacheDict, ArrayCache, VectorCache, MatrixCache, VectorArrayCache, VectorMatrixCache, VectorVectorCache 

include("costs.jl")
export LazyCost

include("utilities.jl")
export SinkhornParameters

include("sinkhorn.jl")
export sinkhorn_dvg, sinkhorn_barycenter

include("sinkhorn_lagrangian.jl")
export sinkhorn_dvg_particles

include("sinkhorn_logseparated.jl")
export sinkhorn_dvg_logseparated, sinkhorn_barycenter_logseparated, sinkhorn_dvg_logsep, sinkhorn_barycenter_logsep

include("sinkhorn_separated.jl")
export sinkhorn_dvg_separated, sinkhorn_barycenter_separated, sinkhorn_dvg_sep, sinkhorn_barycenter_sep

end
