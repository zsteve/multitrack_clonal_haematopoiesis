using Revise
using CSV
using DataFrames
using Base.Threads
using StatsBase
using LinearAlgebra
using Random
using Distributions
using CMAEvolutionStrategy
using ArgParse
using NNlib
using Serialization
using ABM
using Distances
using Random

s = ArgParseSettings()
@add_arg_table! s begin
    "--jobid"
        help = "Job ID (for loading initial condition)"
        arg_type = Int
        default = -1
    "--icfile"
        help = "CSV of initial conditions"
        arg_type = String
        default = "jobs/ics.csv"
    "--data"
        help = "CSV of clones"
        arg_type = String
        default = "data/JS77_filtered_2.txt"
    "--cutoff_size"
        help = "Min. clone size"
        arg_type = Float64
        default = 5.0
    "--tau"
        arg_type = Int
        default = 5
    "--prefactor"
        arg_type = Float64
        default = 1.0
    "--beta_dist"
        arg_type = String
        default = "Gamma(2,1)"
    "--sigma"
        arg_type = Float64
        default = 2.5
    "--maxiter"
        arg_type = Int
        default = 256
    "--popsize"
        arg_type = Int
        default = 24
    "--outdir"
        arg_type = String
        default = "./"
    "--lambda"
    	arg_type = Float64
        default = 1.0
    "--eps"
    	arg_type = Float64
        default = 0.05
    "--suffix"
        arg_type = String
        default = ""
    "--srand"
        arg_type = Int
        default = 0
end
args = parse_args(s)
Random.seed!(args["srand"])

# read data as CSV and convert to Matrix
df = CSV.read(args["data"], DataFrame)
# # New time-point and stage breakdown
# Early1: LT112
# Early2: LT56, ST56
# Mid1: ST28, MPP28
# Mid2: MPP14
# Late: (everything else) ProDC9, CMP9, CLP14, CLP28
df.Stage = map(x -> try Dict(("Day112", "LT") => "1_Early1" ,
		       ("Day56", "LT") => "2_Early2", 
		       ("Day56", "ST") => "2_Early2",
		      ("Day28", "ST") => "3_Mid1", 
		      ("Day28", "MPP") => "3_Mid1", 
		      ("Day14", "MPP") => "4_Mid2", 
	      )[(x.Day, x.HSPC)] catch e "5_Late" end, eachrow(df));
CSV.write(string(splitext(args["data"])[1], "_new_stages.txt"), df)

X_all = df[:, occursin.("n.", names(df)) .& .!occursin.("n.T", names(df))]
clone_sizes = vec(sum(Matrix(X_all); dims = 2))
idx=clone_sizes .>= args["cutoff_size"];

df = df[idx, :];
X_all = Array(X_all[idx, :])
states=["cDC1", "cDC2", "pDC", "Eos", "Mon", "Neu", "B",]
N_states = size(X_all, 2)

# stages 
stages = sort(convert.(String, unique(df.Stage)); rev=false)
N_stages = length(stages)

# Stratify by stage
X = Dict(s => X_all[occursin.(df.Stage, s), :] for s in stages)

# initial distribution
is_unipotent = vec(sum(Array(X_all) .> 0; dims = 2) .== 1)
# p0 = vec(mean(Array(X_all)[is_unipotent, :] .> 0; dims = 1))
# p0 = softmax(2.5*vec(mean(Array(X_all)[is_unipotent, :] .> 0; dims = 1)))
p0 = normalize(ones(size(X_all, 2)), 1)

μ_empirical = Dict(s => hcat([collect(x) for x in eachrow(X[s])]...) for s in stages);
t_stages = Dict(s => args["tau"]*(i-1)+1 for (i, s) in enumerate(stages))
# T = 30
T = args["tau"] * length(stages)

# growth rates
clone_sizes = vcat([log1p.(mean(X[s]; dims = 1)) for (i, s) in enumerate(stages)]...);
dts = [1 - (t_stages[s]-1) / T for s in stages]
g = Dict(s => x / dt for (x, s, dt) in zip(eachrow(clone_sizes), stages, dts))

sim = ABM.ABMSimulation(p0, 
              eval(Meta.parse(args["beta_dist"])),
              states,
              stages, 
              t_stages, 
              T, 
              g,
              ABM.PotentialModel; prefactor = args["prefactor"])
μ_empirical = Dict(s => μ_empirical[s] for s in sim.stages)

function L(x)
    ABM.L(x, μ_empirical, sim; ε = args["eps"], λ = args["lambda"])
end

sim.π0 .= 0
lb_π = one.(sim.π0) * -1
ub_π = one.(sim.π0) * 1
if sim.Ψ isa ABM.RateModel  
    lb_R = sim.Ψ.Rmin*one.(vec(sim.Ψ.R))
    ub_R = sim.Ψ.Rmax*one.(vec(sim.Ψ.R))
    lower = vcat(lb_R, lb_π);
    upper = vcat(ub_R, ub_π);
else
    lb_z = sim.Ψ.zmin*one.(sim.Ψ.z)
    ub_z = sim.Ψ.zmax*one.(sim.Ψ.z)
    lb_v = sim.Ψ.vmin*one.(sim.Ψ.v)[1:end-1]
    ub_v = sim.Ψ.vmax*one.(sim.Ψ.v)[1:end-1]
    lower = vcat(lb_z, lb_v, lb_π);
    upper = vcat(ub_z, ub_v, ub_π);
end

# for generating ICs
# CSV.write("jobs/paramranges.csv", DataFrame(:lb => lower[1:end-length(sim.π0)], :ub => upper[1:end-length(sim.π0)]))
x0=ABM.optvec(sim)
# if jobid > -1, try to load ICs from CSV
if (args["jobid"] > -1) & (splitext(args["icfile"])[end] == ".csv")
    @info "jobid = $(args["jobid"]), initializing from $(args["icfile"])..."
    x0[1:length(ABM.optvec(sim.Ψ))] .= Vector(CSV.read(args["icfile"], DataFrame)[args["jobid"], :])
elseif (splitext(args["icfile"])[end] == ".out")
    @info "initializing from $(args["icfile"])..."
    x0 .= ABM.optvec(deserialize(args["icfile"]))
    args["suffix"] = string(args["suffix"], "_restarted")
else
    @info "jobid = $(args["jobid"]), using default random initialization..."
end;
ABM.from_optvec!(sim, x0)

# parameter trace
param_trace = [] 
function callback(o, y, fvals, perm)
    push!(param_trace, (copy(o.p.mean .+ CMAEvolutionStrategy.sigma(o.p) * y), fvals, perm))
end

result = minimize(L, x0, args["sigma"]; lower = lower, upper = upper, maxiter = args["maxiter"], popsize = args["popsize"], parallel_evaluation = true, callback = callback)

using NPZ
# write trace
npzwrite(string(args["outdir"], "fmedian$(args["suffix"]).npy"), result.logger.fmedian)
serialize(string(args["outdir"], "param_trace$(args["suffix"]).out"), param_trace)
# write initial 
serialize(string(args["outdir"], "sim_init$(args["suffix"]).out"), sim)
# write fit 
sim_fit = copy(sim)
ABM.from_optvec!(sim_fit, xbest(result))
serialize(string(args["outdir"], "sim$(args["suffix"]).out"), sim_fit)
# write data
serialize(string(args["outdir"], "X_empirical$(args["suffix"]).out"), X)
