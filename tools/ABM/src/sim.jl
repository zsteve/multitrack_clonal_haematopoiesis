# Simulation tools 

# β(τ, β0, βmin) = β0*(βmin / β0)^τ
β(τ, β0, βmin) = relu((1-τ)*β0 + τ*βmin)
# β(τ, β0, βmin) = 1 / ((1-τ)*(1/β0) + τ*(1/βmin))

mutable struct ABMSimulation
	π0::Vector{Float64} 
	T::Int
	N_states::Int
	N_stages::Int
	βdist::Distribution
	βmin::Float64
	states::Vector{String}
	stages::Vector{String}
    t_stages::Dict{String, Int}
	w_stages::Dict{String, Float64}
    g::Dict{String, Vector{Float64}}
	Ψ::KineticModel
end

Base.copy(s::ABMSimulation) = ABMSimulation(copy(s.π0),
					   s.T,
					   s.N_states,
					   s.N_stages,
					   s.βdist,
					   s.βmin,
					   copy(s.states),
					   copy(s.stages), 
					   copy(s.t_stages), 
					   copy(s.w_stages),
					   copy(s.g),
					   copy(s.Ψ))

function ABMSimulation(π0::Vector{Float64}, βdist::Distribution, states::Vector{String}, stages::Vector{String}, t_stages::Dict{String, Int}, T::Int, g::Dict{String, Vector{Float64}}, landscape_model; kwargs...)
    βmin = 0.05
	ABMSimulation(π0, 
		      T, 
		      length(π0),
		      length(stages),
		      βdist + βmin, 
		      βmin, 
		      states,
		      stages,
              t_stages, 
              Dict(s => 1/length(stages) for s in stages),
		      g,
              landscape_model(length(π0); kwargs...))
end


function optvec(sim::ABMSimulation)
	vcat(optvec(sim.Ψ), sim.π0)
end

function from_optvec!(sim::ABMSimulation, x)
	M=length(optvec(sim.Ψ))
	from_optvec!(sim.Ψ, x[1:M])
    sim.π0 .= x[M+1:end]
	nothing 
end

function sample_multinomial(n, p)
    return rand(Multinomial(n, p))
end

function _markov_chain_step(x, P, g)
	x_new = [x[i] + (x[i] > 0 ? rand(NegativeBinomial(x[i], g[i])) : x[i]) for i = 1:length(x)] 
	return sum([x_new[i] > 0 ? sample_multinomial(x_new[i], normalize(relu.(P[i, :] .- 1e-9), 1)) : zero.(x) for i = 1:length(x)]) 
end

function _markov_chain_step!(x, x_temp, P, g)
    for i = 1:length(x)
        x[i] += (x[i] > 0 ? rand(NegativeBinomial(x[i], g[i])) : 0)
    end
    fill!(x_temp, 0)
    for i = 1:length(x)
        if x[i] > 0
            x_temp += sample_multinomial(x[i], normalize(relu.(P[i, :] .- 1e-9), 1))
        end
    end
    copy!(x, x_temp)
end

function sample_traj(sim::ABMSimulation, growth_key::String; x0=nothing, β0=nothing, t_init=1, t_final=nothing, growth = true)
	t_final = t_final === nothing ? sim.T : t_final 
	β0 = β0 === nothing ? rand(sim.βdist) : β0
    x0 = (collect(1:sim.N_states) .== (x0 === nothing ? rand(DiscreteNonParametric(1:sim.N_states, softmax(sim.π0))) : x0))*1
    x_init_sample = sample_end(sim, growth_key; t_final = t_init, growth = true, β0 = β0, x0 = x0)
    x = [(collect(1:sim.N_states) .== rand(DiscreteNonParametric(1:sim.N_states, normalize(x_init_sample, 1))))*1, ]
	for t = t_init:t_final-1
		β_curr = β(t / sim.T, β0, sim.βmin)
        g = growth ? exp.(-sim.g[growth_key] / sim.T) : one.(sim.g[growth_key])
		push!(x, _markov_chain_step(x[end], 
				   get_transition_matrix(sim.Ψ, t, sim.T, β_curr),
				   g))
	end
	x
end

function sample_traj_tree(sim::ABMSimulation, growth_key::String; x0=nothing, β0=nothing, t_init=1, t_final=nothing, growth = true)
	t_final = t_final === nothing ? sim.T : t_final 
	β0 = β0 === nothing ? rand(sim.βdist) : β0
    x0 = (collect(1:sim.N_states) .== (x0 === nothing ? rand(DiscreteNonParametric(1:sim.N_states, softmax(sim.π0))) : x0))*1
    x_init_sample = sample_end(sim, growth_key; t_final = t_init, growth = true, β0 = β0, x0 = x0)
    lin = LineageTree((1, (collect(1:sim.N_states) .== rand(DiscreteNonParametric(1:sim.N_states, normalize(x_init_sample, 1))))*1.0, 0))
	for t = t_init:t_final-1
		β_curr = β(t / sim.T, β0, sim.βmin)
        g = growth ? exp.(-sim.g[growth_key] / sim.T) : one.(sim.g[growth_key])
        for leaf_tree in Leaves(lin.root)
            _x = _markov_chain_step(convert.(Int, leaf_tree.state), get_transition_matrix(sim.Ψ, t, sim.T, β_curr), g)
            for i = 1:length(_x)
                while _x[i] > 0
                    push!(leaf_tree.children, LineageTreeNode(-1, (collect(1:sim.N_states) .== i)*1.0, leaf_tree.generation + 1))
                    _x[i] -= 1
                end
            end
        end
		# push!(x, _markov_chain_step(x[end], 
		# 		   get_transition_matrix(sim.Ψ, t, sim.T, β_curr),
		# 		   g))
	end
	lin
end


function sample_end(sim::ABMSimulation, growth_key::String; x0=nothing, β0=nothing, t_init=1, t_final=nothing, growth = true)
	t_final = t_final === nothing ? sim.T : t_final 
    x0 = x0 === nothing ? (collect(1:sim.N_states) .== rand(DiscreteNonParametric(1:sim.N_states, softmax(sim.π0))))*1 : copy(x0)
	β0 = β0 === nothing ? rand(sim.βdist) : β0
	x = x0
    x_temp = similar(x)
    P_cache = similar(ABM.optvec(sim.Ψ), sim.N_states, sim.N_states)
    g = growth ? exp.(-sim.g[growth_key] / sim.T) : one.(sim.g[growth_key])
	for t = t_init:t_final-1
		β_curr = β(t / sim.T, β0, sim.βmin)
        P_cache .= get_transition_matrix(sim.Ψ, t, sim.T, β_curr)
		_markov_chain_step!(x, x_temp, P_cache, g)
	end
	x
end

function sample_timepoint(sim::ABMSimulation, t::Int, growth_key::String; N = 128)
	clones = zeros(Int64, sim.N_states, N)
    for i = 1:N
        β0 = rand(sim.βdist)
        x0 = sample_end(sim, growth_key; t_final = t, growth = true, β0 = β0)
        clones[:, i] .= sample_end(sim,
                                  growth_key; 
                                  x0 = (collect(1:sim.N_states) .== rand(DiscreteNonParametric(1:sim.N_states, normalize(x0, 1))))*1,
                                  β0 = β0, t_init = t)
    end
	return clones
end

function sample(sim::ABMSimulation; N = 128)
    clones_all = Dict(s => sample_timepoint(sim, sim.t_stages[s], s; N = N) for s in sim.stages)
	return clones_all
end

function fate_map(sim::ABMSimulation, growth_key::String; kwargs...)
    normalize(mean(fate_cloud_map(sim, growth_key; kwargs...); dims = 1), 1)'
end

function fate_cloud_map(sim::ABMSimulation, growth_key::String; x0 = nothing, β0 = nothing, t_init = 1, N = 256, growth = false)
    x0 = x0 === nothing ? (collect(1:sim.N_states) .== rand(DiscreteNonParametric(1:sim.N_states, softmax(sim.π0))))*1 : (collect(1:sim.N_states) .== x0)*1
	β0 = β0 === nothing ? rand(sim.βdist) : β0
    hcat([ABM.sample_end(sim, growth_key; x0 = x0, β0 = β0, t_init = t_init, growth = growth) for _ in 1:N]...)'
end

function traj_fate_map(sim::ABMSimulation, traj, growth_key::String, β0::Float64; t_init = 1, kwargs...)
    fates = []
    times = []
    for (i, x) in enumerate(traj)
        try
            t = t_init-1+i
            tmp=hcat([fate_map(sim, growth_key; x0=j, β0=β0, t_init=t, kwargs...)*x[j] for j in findall(x .> 0)]...)'
            # remove zero rows in case
            tmp=tmp[vec(sum(tmp; dims = 2)) .> 0, :]
            push!(fates, tmp)
            push!(times, fill(t, size(tmp, 1)))
        catch
        end
    end
    return vcat(fates...), vcat(times...)
end

function traj_fate_cloud_map(sim::ABMSimulation, traj, growth_key::String, β0::Float64; t_init = 1, kwargs...)
    fates = []
    times = []
    for (i, x) in enumerate(traj)
        try
            t = t_init-1+i
            tmp=vcat([fate_cloud_map(sim, growth_key; x0=j, β0=β0, t_init=t, kwargs...)*x[j] for j in findall(x .> 0)]...)
            # remove zero rows in case
            tmp=tmp[vec(sum(tmp; dims = 2)) .> 0, :]
            push!(fates, tmp)
            push!(times, fill(t, size(tmp, 1)))
        catch
        end
    end
    return vcat(fates...), vcat(times...)
end
