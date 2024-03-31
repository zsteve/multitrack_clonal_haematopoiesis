module ABM

using Distributions
using LinearAlgebra
using Base.Threads
using DataInterpolations
using ExponentialUtilities
using OptimalTransport
using Distances
using NNlib
using AbstractTrees

include("landscape.jl")
include("sim.jl")
include("lineagetree.jl")

greet() = print("ABM module loaded")

function d(μ, ν; ε = 0.05, λ = 5.0)
	source = log1p.(μ)
	source = relu.(source ./ sum(source; dims = 1))
	target = log1p.(ν)
	target = relu.(target ./ sum(target; dims = 1))
    dist = SqEuclidean()
	cμν = Distances.pairwise(dist, source, target)
	cμμ = Distances.pairwise(dist, source, source)
	cνν = Distances.pairwise(dist, target, target)
	m = size(μ, 2)
	n = size(ν, 2)
    # p = fill(1/m, m) 
    p = normalize(log1p.(vec(sum(μ; dims = 1))), 1)
    # q = fill(1/n, n) 
    q = normalize(log1p.(vec(sum(ν; dims = 1))), 1)
	# cmean = mean([mean(cμν), mean(cμμ), mean(cνν)])
	# OptimalTransport.sinkhorn_divergence_unbalanced(fill(1/m, m), fill(1/n, n), cμν, cμμ, cνν, λ, ε; maxiter = 1_000, atol = 1e-9, rtol = 1e-9)
	OptimalTransport.sinkhorn_divergence_unbalanced(p, q, cμν, cμμ, cνν, λ, ε; maxiter = 1_000, atol = 1e-9, rtol = 1e-9)
	# OptimalTransport.sinkhorn_divergence(fill(1/m, m), fill(1/n, n), cμν, cμμ, cνν, ε; maxiter = 1_000, atol = 1e-4, rtol = 1e-4)
	# EMD.emd2(fill(1/m, m), fill(1/n, n), cμν)
end

function d_all(μ, ν; kwargs...)
	@assert keys(μ) == keys(ν)
	Dict(s => d(μ[s], ν[s]; kwargs...) for s in keys(μ))
end

function reg(sim::ABMSimulation)
    return norm(sim.Ψ.z, 2)^2 + norm(sim.Ψ.v, 2)^2 
end

function F(x, μ, sim::ABMSimulation; w_reg = 1e-5, kw_sample = Dict(), kwargs...)
	from_optvec!(sim, x)
	d = d_all(sample(sim; kw_sample...), μ; kwargs...)
    return sum([sim.w_stages[s] * d[s] for s in keys(d)]) + w_reg*reg(sim)
end

# parallel evaluation of population fitness 
function L(x, μ, sim::ABMSimulation; kwargs...)
	out = similar(x, size(x, 2))
	@threads for i = 1:size(x, 2)
	    out[i] = F(x[:, i], μ, copy(sim); kwargs...)
	end
	out
end

end # module ABM
