abstract type AbstractPropagatorAlg end
struct BlockPropagatorAlg <: AbstractPropagatorAlg end

function scrambling_map(H_main, H_res, H_total, measurements, ψres,
        hamiltonian, t::Number, ::BlockPropagatorAlg)
    ρ_res = density_matrix(ψres)
    U = propagator(t, hamiltonian)
    measurements_t = map(Base.Fix1(operator_time_evolution, U), measurements)
    eff_measurements = map(
        mt -> effective_measurement(mt, ρ_res, H_main, H_res, H_total),
        measurements_t)
    return reduce(vcat, (vec(m)' for m in eff_measurements))
end

struct PureStatePropagatorAlg <: AbstractPropagatorAlg
    krylov_dim::Int
    tol::Float64
end
function PureStatePropagatorAlg(; krylov_dim = 200, tol = 1e-6)
    PureStatePropagatorAlg(krylov_dim, tol)
end
function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
        hamiltonian, t::Number, alg::PureStatePropagatorAlg)
    iH = -im .* hamiltonian
    N = dim(H_total)
    N_main = dim(H_main)
    Ks = KrylovSubspace{ComplexF64}(N, alg.krylov_dim)
    e_j = zeros(ComplexF64, N_main)
    U = stack(1:N_main) do n
        fill!(e_j, 0)
        e_j[n] = 1.0
        ψtot = tensor_product((e_j, ψres), (H_main, H_res) => H_total)
        arnoldi!(Ks, iH, ψtot; tol = alg.tol)
        expv(t, Ks)
    end
    stack(op -> vec(U' * Diagonal(op) * U), measurements)'
end

function scrambling_map(H_main, H_res, H_total, measurements,
        ψres, hamiltonian, t::AbstractArray, alg)
    mapreduce(
        ti -> scrambling_map(
            H_main, H_res, H_total, measurements, ψres, hamiltonian, ti, alg),
        vcat,
        t)
end

function scrambling_map(sys::QuantumDotSystem, measurements, ψres,
        hamiltonian, t, alg = PureStatePropagatorAlg())
    scrambling_map(
        sys.H_main, sys.H_res, sys.H_total, measurements, ψres, hamiltonian, t, alg)
end