using LinearAlgebra: norm
using WassersteinDictionaries
using Test

const d = 2     # dimension
const n = 64    # grid size

x₁ = collect(range(0,1,length=n))
x₂ = collect(range(0,1,length=n))

x = [x₁, x₂]; y = [x₁, x₂];

h₁ = 1/n
h₂ = 1/n

S = 4           # number of distributions

c = WassersteinDictionaries.get_cost_matrix_separated(x, y, d)  # cost

α = [zeros(n,n) for _ in 1:S]  # input histograms
μ₀ₑ = zeros(n,n)               # exact barycenter


function p₁(x) 
    exp(-norm(x-[0.15; 0.15])^2/(2*0.04^2))
end
function p₂(x)
    exp(-norm(x-[0.85; 0.85])^2/(2*0.04^2))
end
function p₃(x)
    exp(-norm(x-[0.8; 0.3])^2/(2*0.04^2))
end
function p₄(x)
    exp(-norm(x-[0.20; 0.70])^2/(2*0.04^2))
end
function m_(x)
    exp(-norm(x-[0.5; 0.5])^2/(2*0.04^2))
end

for i in 1:n
    for j in 1:n
        α[1][i,j] = p₁([x₁[i]; x₂[j]])
        α[2][i,j] = p₂([x₁[i]; x₂[j]])
        α[3][i,j] = p₃([x₁[i]; x₂[j]])
        α[4][i,j] = p₄([x₁[i]; x₂[j]])
        μ₀ₑ[i,j] = m_([x₁[i]; x₂[j]])
    end
end

for s in 1:S 
    α[s] ./= sum(α[s])
end
μ₀ₑ ./= sum(μ₀ₑ);

log_α = [ log.(α[s]) for s in 1:S ];

ε = 5e-3
k = WassersteinDictionaries.get_gibbs_matrix(c, ε)

SP = SinkhornParameters(128, ε)
SPB = SinkhornParameters(256, ε)
caches = ( MC = MatrixCache(n), VC = VectorCache(n) )

SP.tol = 1e-9 * h₁ * h₂
SPB.tol = 1e-9 * h₁ * h₂

SP.update_potentials = false

@testset "Distances" begin
    SP.debias = false
    SP.averaged_updates = false
    @test abs(sinkhorn_dvg_sep(α[1], α[2], k, SP, caches) - 0.49) / 0.49 < 2 * ε
    @test abs(sinkhorn_dvg_logsep(log_α[1], log_α[2], c, SP, caches) - 0.49) / 0.49 < 2 * ε
    @test sinkhorn_dvg_logsep(log_α[1], log_α[2], c, SP, caches) ≈ sinkhorn_dvg_sep(α[1], α[2], k, SP, caches) atol = 1e-12

    SP.averaged_updates = true
    @test abs(sinkhorn_dvg_sep(α[1], α[2], k, SP, caches) - 0.49) / 0.49 < 2 * ε
    @test abs(sinkhorn_dvg_logsep(log_α[1], log_α[2], c, SP, caches) - 0.49) / 0.49 < 2 * ε
    @test sinkhorn_dvg_logsep(log_α[1], log_α[2], c, SP, caches) ≈ sinkhorn_dvg_sep(α[1], α[2], k, SP, caches) atol = 1e-12

    SP.debias = true
    @test abs(sinkhorn_dvg_sep(α[1], α[2], k, SP, caches) - 0.49) / 0.49 < 2 * ε^2
    @test abs(sinkhorn_dvg_logsep(log_α[1], log_α[2], c, SP, caches) - 0.49) / 0.49 < 2 * ε^2
    @test sinkhorn_dvg_logsep(log_α[1], log_α[2], c, SP, caches) ≈ sinkhorn_dvg_sep(α[1], α[2], k, SP, caches) atol = 1e-12

    SP.averaged_updates = false
    @test abs(sinkhorn_dvg_sep(α[1], α[2], k, SP, caches) - 0.49) / 0.49 < 2 * ε^2
    @test abs(sinkhorn_dvg_logsep(log_α[1], log_α[2], c, SP, caches) - 0.49) / 0.49 < 2 * ε^2
    @test sinkhorn_dvg_logsep(log_α[1], log_α[2], c, SP, caches) ≈ sinkhorn_dvg_sep(α[1], α[2], k, SP, caches) atol = 1e-12
end

@testset "Barycenters" begin
    λ₀ = ones(S)/S

    SPB.debias = true
    SPB.averaged_updates = false
    μ₀ = sinkhorn_barycenter_sep(λ₀, α, k, SPB, caches)
    log_μ₀ = sinkhorn_barycenter_logsep(λ₀, log_α, c, SPB, caches)

    @test norm(μ₀ - μ₀ₑ,1) / norm(μ₀ₑ,1) < 1e-3
    @test norm(exp.(log_μ₀) - μ₀ₑ,1) / norm(μ₀ₑ,1) < 1e-3
    @test norm(μ₀ - exp.(log_μ₀),1) / norm(μ₀,1) < 1e-9

    SPB.averaged_updates = true

    μ₀ = sinkhorn_barycenter_sep(λ₀, α, k, SPB, caches)
    log_μ₀ = sinkhorn_barycenter_logsep(λ₀, log_α, c, SPB, caches)

    @test norm(μ₀ - μ₀ₑ,1) / norm(μ₀ₑ,1) < 1e-3
    @test norm(exp.(log_μ₀) - μ₀ₑ,1) / norm(μ₀ₑ,1) < 1e-3
    @test norm(μ₀ - exp.(log_μ₀),1) / norm(μ₀,1) < 1e-5 # symmetric updates do not perform as well as wished

end

