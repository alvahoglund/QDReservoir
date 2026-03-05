## Standard time evolution
propagator(t, hamiltonian) = cis(-t * Matrix(hamiltonian))

operator_time_evolution(propagator, operator) = propagator' * operator * propagator
operator_time_evolution(operator, t, hamiltonian) = operator_time_evolution(propagator(t, hamiltonian), operator)

state_time_evolution(propagator, ρ) = propagator * ρ * propagator'
state_time_evolution(ρ, t, hamiltonian) = state_time_evolution(propagator(t, hamiltonian), ρ)
