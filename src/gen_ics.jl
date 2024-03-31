using QuasiMonteCarlo
using ArgParse
using CSV
using DataFrames 

s = ArgParseSettings()
@add_arg_table! s begin
    "param_ranges"
        help = "CSV of parameter ranges"
        arg_type = String
        default = "jobs/paramranges.csv"
    "--N"
        help = "number of samples"
        arg_type = Int
        default = 100
    "--outfile"
        help = "output file"
        arg_type = String
        default = "jobs/ics.csv"
end
args = parse_args(s)
paramranges = CSV.read(args["param_ranges"], DataFrame)
samples = QuasiMonteCarlo.sample(args["N"], paramranges.lb, paramranges.ub, LatinHypercubeSample())
CSV.write(args["outfile"], DataFrame(samples', :auto))

