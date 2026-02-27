abstract type AbstractPropagatorAlg end
struct FullPropagatorAlg <: AbstractPropagatorAlg end
struct BlockPropagatorAlg <: AbstractPropagatorAlg end

function scrambling_map(qd_system::QuantumDotSystem, measurements, ψres, hamiltonian_total, t, ::FullPropagatorAlg)
    ρ_res = if ψres isa AbstractVector
        ψres * ψres'
    else
        ψres
    end
    prop = propagator(t, hamiltonian_total, qd_system.qn_total, qd_system.H_total)
    process_measurements = op -> effective_measurement(
        operator_time_evolution(prop, sparse(op)), ρ_res, qd_system
    )
    eff_measurements = map(process_measurements, measurements)
    scrambling_map = vcat([vec(m)' for m in eff_measurements]...)
    return scrambling_map
end


function scrambling_map(qd_system, t::Number, measurement_types=charge_measurements, alg=BlockPropagatorAlg(); seed=nothing)
    hams = hamiltonians(qd_system, seed)
    hamiltonian_total = matrix_representation(hams.hamiltonian_total, qd_system.H_total)
    ρ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)
    measurements = map(op -> matrix_representation(op, qd_system.H_total), measurement_types(qd_system))
    return scrambling_map(qd_system, measurements, ρ_res, hamiltonian_total, t, alg)
end

function scrambling_map(qd_system, t_list::AbstractArray, measurement_types, alg=BlockPropagatorAlg(); seed=nothing)
    hams = hamiltonians(qd_system, seed)
    hamiltonian_total = matrix_representation(hams.hamiltonian_total, qd_system.H_total)
    ψ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)
    ρ_res = ψ_res * ψ_res'
    measurements = map(op -> matrix_representation(op, qd_system.H_total), measurement_types(qd_system))
    scrambling_maps = [scrambling_map(qd_system, measurements, ρ_res, hamiltonian_total, t, alg) for t in t_list]
    return vcat(scrambling_maps...)
end


function pauli_string_map(qd_system)
    measurements = pauli_strings(qd_system)
    ind = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)
    measurements = map(op -> op[ind, ind], measurements)
    pauli_string_map = vcat([vec(m)' for m in measurements]...)
    return pauli_string_map
end

function scrambling_map(qd_system::QuantumDotSystem, measurements, ψres, hamiltonian_total, t, ::BlockPropagatorAlg)
    ρ_res = if ψres isa AbstractVector
        ψres * ψres'
    else
        ψres
    end
    prop = propagator(t, hamiltonian_total, qd_system.qn_total, qd_system.H_total)
    inds = indices(qd_system.qn_total, qd_system.H_total)
    prop_block = Matrix(prop[inds, inds])
    function process_measurements(op)
        future_op_block = operator_time_evolution(prop_block, Diagonal(diag(op)[inds]))
        future_op = spzeros(Complex{Float64}, dim(qd_system.H_total), dim(qd_system.H_total))
        future_op[inds, inds] = future_op_block
        effective_measurement(future_op, ρ_res, qd_system)
    end
    eff_measurements = map(process_measurements, measurements)
    scrambling_map = vcat([vec(m)' for m in eff_measurements]...)
    return scrambling_map
end

struct PureStatePropagatorAlg <: AbstractPropagatorAlg
    krylov_dim::Int
    tol::Float64
end
PureStatePropagatorAlg(; krylov_dim=200, tol=1e-6) = PureStatePropagatorAlg(krylov_dim, tol)
function scrambling_map(sys::QuantumDotSystem, measurements, ψres::AbstractVector, hamiltonian, t, alg::PureStatePropagatorAlg)
    iH = -im .* hamiltonian
    N = dim(sys.H_total_qn)
    N_main = dim(sys.H_main_qn)
    Ks = KrylovSubspace{ComplexF64}(N, alg.krylov_dim)
    e_j = zeros(ComplexF64, N_main)
    U = zeros(ComplexF64, N, N_main)
    U = stack(1:N_main) do n
        fill!(e_j, 0)
        e_j[n] = 1.0
        ψtot = generalized_kron((e_j, ψres), (sys.H_main_qn, sys.H_reservoir_qn) => sys.H_total_qn)
        arnoldi!(Ks, iH, ψtot; tol=alg.tol)
        expv(t, Ks)
    end
    stack(op -> vec(U' * Diagonal(op) * U), measurements)'
end

