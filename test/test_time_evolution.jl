
@testset "State and operator evolution" begin
    using FermionicHilbertSpaces, LinearAlgebra, Random

    ρ = rand(ComplexF64, 4, 4) + hc
    ρ = ρ ./ tr(ρ)
    ham = (rand(ComplexF64, 4, 4) + hc) / 2
    op = (rand(ComplexF64, 4, 4) + hc) / 2
    t = 1.0

    ρt = state_time_evolution(ρ, t, ham)
    exp_value_ρ = expectation_value(ρt, op)

    op_t = operator_time_evolution(op, t, ham)
    exp_value_op = expectation_value(ρ, op_t)

    @test exp_value_op ≈ exp_value_ρ
end
@testset "Effective measurements & time evolution" begin
    ## Initialize system
    quantum_dot_system = tight_binding_system(2, 3, 1)
    seed = 2
    hams = hamiltonians(quantum_dot_system, seed)
    reservoir_state = ground_state(hams.hamiltonian_reservoir, quantum_dot_system.H_reservoir, quantum_dot_system.qn_reservoir)

    initial_states = [def_state(triplet_plus, quantum_dot_system.H_main, quantum_dot_system.f),
        def_state(singlet, quantum_dot_system.H_main, quantum_dot_system.f),
        random_product_state(quantum_dot_system),
        random_separable_state(3, quantum_dot_system)]
    ρres = reservoir_state * reservoir_state'
    total_states = map(initial_state -> tensor_product((initial_state, ρres), (quantum_dot_system.H_main, quantum_dot_system.H_reservoir) => quantum_dot_system.H_total), initial_states)
    measurements = map(op -> matrix_representation(op, quantum_dot_system.H_total), charge_measurements(quantum_dot_system))

    t = 10
    ham_total = matrix_representation(hams.hamiltonian_total, quantum_dot_system.H_total)

    time_evolved_states = map(total_state -> state_time_evolution(total_state, t, ham_total, quantum_dot_system.H_total, quantum_dot_system.qn_total), total_states)
    time_evolved_measurements = map(measurement -> operator_time_evolution(measurement, t, ham_total, quantum_dot_system.qn_total, quantum_dot_system.H_total), measurements)
    effective_measurements = map(measurement -> effective_measurement(measurement, ρres, quantum_dot_system), time_evolved_measurements)
    scrambling = scrambling_map(quantum_dot_system, t, charge_measurements, QuantumDotReservoir.FullPropagatorAlg(); seed)

    ind = FermionicHilbertSpaces.indices(quantum_dot_system.H_main_qn, quantum_dot_system.H_main)

    for i in eachindex(initial_states), j in eachindex(measurements)
        exp_val_1 = expectation_value(time_evolved_states[i], measurements[j])
        exp_val_2 = expectation_value(total_states[i], time_evolved_measurements[j])
        exp_val_3 = expectation_value(initial_states[i][ind, ind], effective_measurements[j])
        exp_val_4 = (scrambling*vec(initial_states[i][ind, ind]))[j]
        @test exp_val_1 ≈ exp_val_2 ≈ exp_val_3 ≈ exp_val_4
    end
end

