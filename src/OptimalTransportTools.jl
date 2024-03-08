module OptimalTransportTools

using ForwardDiff
using LinearAlgebra
using Distances
using LogExpFunctions
using Base.Threads

import SimpleSolvers as SS

using Printf

include("caches.jl")
export CacheDict, ArrayCache, VectorCache, MatrixCache

include("costs.jl")
export LazyCost

include("utilities.jl")
export SinkhornParameters

#include("sinkhorn.jl")
#export sinkhorn_dvg, sinkhorn_barycenter

#include("sinkhorn_lagrangian.jl")
#export sinkhorn_dvg_particles

include("sinkhorn_logseparated.jl")
export sinkhorn_dvg_logseparated, sinkhorn_barycenter_logseparated, sinkhorn_dvg_logsep, sinkhorn_barycenter_logsep

include("sinkhorn_separated.jl")
export sinkhorn_dvg_separated, sinkhorn_barycenter_separated, sinkhorn_dvg_sep, sinkhorn_barycenter_sep

end
