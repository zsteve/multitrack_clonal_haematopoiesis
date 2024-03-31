# Discrete landscape model 

vech_strict(A) = A[triu(trues(size(A)), 1)]

function unvech_strict(x)
    # length(x) = n(n-1)/2 for nxn matrix
    n = convert(Int, (1 + sqrt(1 + 8*length(x)))/2)
    A = similar(x, n, n)
    fill!(A, -Inf)
    # diagonal of zeros
    A[diagind(A)] .= 0
    A[triu(trues(size(A)), 1)] .= x
    max.(A, A')
end

abstract type KineticModel end 

struct RateModel <: KineticModel
	n::Int
    R::Vector{Float64}
	Rmin::Float64
	Rmax::Float64
	prefactor::Float64
end

mutable struct PotentialModel <: KineticModel
    n::Int
    z::Vector{Float64}
    v::Vector{Float64}
    v_proj_mat::Matrix{Float64}
    zmin::Float64
    zmax::Float64
    vmin::Float64
    vmax::Float64
    prefactor::Float64
end

Base.copy(l::RateModel)=RateModel(l.n, copy(l.R), l.Rmin, l.Rmax, l.prefactor)
Base.copy(l::PotentialModel)=PotentialModel(l.n, copy(l.z), copy(l.v), copy(l.v_proj_mat), l.zmin, l.zmax, l.vmin, l.vmax, l.prefactor)

cutoff(x) = max(min(x, 1), 0)

function optvec(l::RateModel)
    vec(l.R)
end

function from_optvec!(l::RateModel, x)
	@assert length(optvec(l)) == length(x)
    l.R .= reshape(x, size(l.R))
	nothing 
end

function RateModel(n; Rmin = 0.0, Rmax = 0.95, prefactor = 1.0)
    R = R_init_rand(n, Rmin, Rmax)
	RateModel(n, 
          R, 
		  Rmin, Rmax,
		  prefactor)
end

function PotentialModel(n; zmin = 0.0, zmax = 10.0, vmin = -5.0, vmax = 5.0, prefactor = 1.0)
    A = ones(n, n)
    A -= Diagonal(diag(ones(n, n) * A))
    v_proj_mat = Matrix(qr(A[:, 1:end-1]).Q)
    PotentialModel(n, 
          z_init_rand(n, zmin, zmax), 
          v_proj_mat * v_init_rand(n, vmin, vmax), 
          v_proj_mat, 
          zmin, zmax,
          vmin, vmax, prefactor)
end

function _get_rate_matrix(R, D)
	R_upper = R[1:(length(R) ÷ 2)]
	R_lower = R[(length(R)÷2 + 1):end]
	A = (triu(unvech_strict(R_upper)) + triu(unvech_strict(R_lower))').^(1/D)
	A[diagind(A)] .= 0
    A .-= Diagonal(vec(sum(A; dims = 2)))
    A
end


function _get_rate_matrix!(A, R, D)
    function unvech_strict!(A, x; lower = false)
        n = convert(Int, (1 + sqrt(1 + 8*length(x)))/2)
        k=1
        for j = 2:n
            for i = 1:(j-1)
                if lower
                    A[j, i] = x[k]
                else
                    A[i, j] = x[k]
                end
                k += 1
            end
        end
    end
	idx_upper = 1:(length(R) ÷ 2)
    idx_lower = (length(R)÷2 + 1):length(R)
	# A = (triu(unvech_strict(R_upper)) + triu(unvech_strict(R_lower))').^(1/D)
    fill!(A, 0)
    unvech_strict!(A, R[idx_upper]; lower = false)
    unvech_strict!(A, R[idx_lower]; lower = true)
    @. A = A^(1/D)
    A .-= Diagonal(vec(sum(A; dims = 2)))
    A
end

function get_rate_matrix(l::RateModel, D, t)
    l.prefactor*_get_rate_matrix(l.R, D)
end

function get_rate_matrix!(A, l::RateModel, D, t)
    _get_rate_matrix!(A, l.R, D)
    rmul!(A, l.prefactor)
end

function get_transition_matrix(l::RateModel, t, T, D)
	A = get_rate_matrix(l, D, t)/T
    exponential!(A)
end

function get_transition_matrix!(A, l::RateModel, t, T, D)
	get_rate_matrix!(A, l, D, t)
    rmul!(A, 1/T)
    exponential!(A)
end

function optvec(l::PotentialModel)
    vcat(l.z, l.v_proj_mat' * l.v)
end

function from_optvec!(l::PotentialModel, x)
    @assert length(optvec(l)) == length(x)
    l.z = x[1:length(l.z)]
    l.v = l.v_proj_mat * x[length(l.z)+1:end]
    nothing 
end

function get_energetic_barrier(l::PotentialModel)
    _get_energetic_barrier(unvech_strict(l.z), l.v)
end

function _get_energetic_barrier(z, v)
    V = max.(v, v') .+ z
    (V .- v)
end

function _get_rate_matrix(z, v, D)
    A = exp.(-_get_energetic_barrier(z, v)/D)
    A[diagind(A)] .= 0
    A - Diagonal(vec(sum(A; dims = 2)))
end

function get_rate_matrix(l::PotentialModel, D, t)
    l.prefactor*_get_rate_matrix(unvech_strict(l.z), l.v, D)
end

function get_transition_matrix(l::PotentialModel, t, T, D)
    exp(get_rate_matrix(l, D, t)/T)
end

z_init_rand(N_states, zmin, zmax; dist = Gamma(2, 1)) = rand(truncated(dist, zmin, zmax), N_states*(N_states-1)>>1)
v_init_rand(N_states, vmin, vmax; dist = Normal(0, 1)) = rand(truncated(dist, vmin, vmax), N_states-1)
R_init_rand(N_states, Rmin, Rmax; dist = Dirac(0.01)) = rand(truncated(dist, Rmin, Rmax), N_states*(N_states-1))
