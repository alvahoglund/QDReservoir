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
        ψtot = generalized_kron((e_j, ψres), (H_main, H_res) => H_total)
        arnoldi!(Ks, iH, ψtot; tol = alg.tol)
        expv(t, Ks)
    end
    stack(op -> vec(U' * Diagonal(op) * U), measurements)'
end

struct PureStateSteppingPropagatorAlg{T1 <: Number, T2 <: Number} <: AbstractPropagatorAlg
    krylov_dim::Int
    tol::Float64
    step_size::T1
    step_size_const::T2
end
function PureStateSteppingPropagatorAlg(;
        krylov_dim = 100, tol = 1e-6, step_size = 0, step_size_const = 1)
    return PureStateSteppingPropagatorAlg(
        krylov_dim, tol, step_size, step_size_const)
end

function get_step_size(ham, alg::PureStateSteppingPropagatorAlg)
    if alg.step_size != 0
        return alg.step_size
    end
    λ_large, _ = eigsolve(ham, rand(ComplexF64, size(ham, 1)), 1, :LR)
    λ_small, _ = eigsolve(ham, rand(ComplexF64, size(ham, 1)), 1, :SR)
    bandwidth = real(λ_large[1]) - real(λ_small[1])
    step_size = alg.step_size_const * (alg.krylov_dim) / bandwidth
    return step_size
end

function krylov_step(Ks, iH, ψ::AbstractVector, dt, tol)
    arnoldi!(Ks, iH, ψ; tol = tol)
    return expv(dt, Ks)
end

function expv_stepped(
        iH, ψtot::AbstractVector, t::Number, alg::PureStateSteppingPropagatorAlg, step_size)
    Ks = KrylovSubspace{ComplexF64}(length(ψtot), alg.krylov_dim)

    nbr_steps = ceil(Int, abs(t) / min(step_size, abs(t)))
    dt = t / nbr_steps
    ψ_step = ψtot
    for _ in 1:nbr_steps
        ψ_step = krylov_step(Ks, iH, ψ_step, dt, alg.tol)
    end
    return ψ_step
end

function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
        hamiltonian, t::Number, alg::PureStateSteppingPropagatorAlg)
    iH = -im .* hamiltonian

    step_size = get_step_size(hamiltonian, alg)
    N_main = dim(H_main)
    e_j = zeros(ComplexF64, N_main)
    U = stack(1:N_main) do n
        fill!(e_j, 0)
        e_j[n] = 1.0
        ψtot = generalized_kron((e_j, ψres), (H_main, H_res) => H_total)
        expv_stepped(iH, ψtot, t, alg, step_size)
    end
    stack(op -> vec(U' * Diagonal(op) * U), measurements)'
end

struct DiagPropagatorAlg <: AbstractPropagatorAlg end

function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
        hamiltonian, t::Number, ::DiagPropagatorAlg)
    N_main = dim(H_main)
    F = eigen(Hermitian(Matrix(hamiltonian)))
    phases = exp.(-im .* F.values .* t)
    e_j = zeros(ComplexF64, N_main)
    U = stack(1:N_main) do n
        fill!(e_j, 0)
        e_j[n] = 1.0
        ψtot = generalized_kron((e_j, ψres), (H_main, H_res) => H_total)
        F.vectors * (phases .* (F.vectors' * ψtot))
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