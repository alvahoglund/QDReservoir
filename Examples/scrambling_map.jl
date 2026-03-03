##############
using QDReservoir, LinearAlgebra
import QDReservoir as QDR
nbr_dots_main = 2
nbr_dots_res = 6
qn_reservoir = 3
sys = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
seed = 2
hams = QDR.matrix_representation_hams(hamiltonians(sys, seed), sys)
ψres = ground_state(hams.hamiltonian_reservoir, QDR.ArnoldiAlg())
measurements = map(op -> matrix_representation(op, sys.H_total_qn), charge_measurements(sys.grid.total, sys.f))

@time sm_pure = scrambling_map(sys, measurements, ψres, hams.hamiltonian_total, t, QDR.PureStatePropagatorAlg(; krylov_dim=200, tol=1e-6));
## Check convergence of the scrambling map with respect to the Krylov dimension!
sm_pures = [scrambling_map(sys, measurements, ψres, hams.hamiltonian_total, t, QDR.PureStatePropagatorAlg(; krylov_dim, tol=1e-6)) for krylov_dim in [100, 200, 300]]
norm.(diff(sm_pures))
##
@profview sm_pure = scrambling_map(sys, measurements, ψres, hams.hamiltonian_total, t, QDR.PureStatePropagatorAlg(; krylov_dim=200, tol=1e-6));
##
@time reservoir_state = ground_state(hams.hamiltonian_reservoir, QDR.ExactDiagonalizationAlg())
@time measurements_total = map(op -> matrix_representation(op, sys.H_total_qn), charge_measurements(sys.grid.total, sys.f))
@time sm_block = scrambling_map(sys, measurements_total, ψres, hams.hamiltonian_total, t, QDR.BlockPropagatorAlg());
sm_block ≈ sm_pure
#@time sm_krylov = scrambling_map(quantum_dot_system, measurements, ρres, ham_total, t, QDR.KrylovPropagatorAlg());
# @profview sm_pure = scrambling_map(quantum_dot_system, measurements, ψres, ham_total, t, QDR.PureStatePropagatorAlg(; krylov_dim=200, tol=1e-6));
##
# ind = indices(sys.H_main_qn, sys.H_main)

# total_states = map(initial_state -> tensor_product((initial_state, ρres), (sys.H_main, sys.H_reservoir) => sys.H_total), initial_states);
initial_states = [def_state(triplet_plus, sys.H_main, sys.f),
    def_state(singlet, sys.H_main, sys.f),
    random_product_state(sys),
    random_separable_state(3, sys)]
# time_evolved_states = map(total_state -> state_time_evolution(total_state, t, ham_total, quantum_dot_system.H_total, quantum_dot_system.qn_total), total_states)
# time_evolved_measurements = map(measurement -> operator_time_evolution(measurement, t, ham_total, quantum_dot_system.qn_total, quantum_dot_system.H_total), measurements)
# effective_measurements = map(measurement -> effective_measurement(measurement, ρres, quantum_dot_system), time_evolved_measurements)
@profview sm = scrambling_map(sys, measurements, reservoir_state, ham_total, t, QuantumDotReservoir.FullPropagatorAlg());
@profview sm = scrambling_map(sys, measurements, reservoir_state, ham_total, t, QuantumDotReservoir.BlockPropagatorAlg());
@profview sm = scrambling_map(sys, measurements, reservoir_state, ham_total, t, QuantumDotReservoir.KrylovPropagatorAlg());

@time measured_values = map(m -> expectation_value(time_evolved_states[3], m), measurements)
if nbr_dots_res ≥ 6
    reshape(inv(sm) * measured_values, 4, 4) ≈ initial_states[3][ind, ind]
end
###############