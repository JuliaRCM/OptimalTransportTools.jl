module WassersteinDictionaries

using ForwardDiff
using LinearAlgebra
using Distances
using SimpleSolvers
using LogExpFunctions

using Printf

include("caches.jl")
export CacheDict, ArrayCache, VectorCache, MatrixCache, VectorArrayCache, VectorMatrixCache, VectorVectorCache 

include("costs.jl")
export LazyCost

include("utilities.jl")
export SinkhornParameters

include("sinkhorn_lagrangian.jl")
export sinkhorn_dvg_particles

include("sinkhorn_logseparated.jl")
export sinkhorn_dvg_logseparated, sinkhorn_barycenter_logseparated

include("sinkhorn_separated.jl")
export sinkhorn_dvg_separated, sinkhorn_barycenter_separated

include("solvers.jl")
export mysolve!

end
