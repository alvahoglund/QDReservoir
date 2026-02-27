propagator(t, hamiltonian) = cis(-t * Matrix(hamiltonian))

function propagator(t, hamiltonian, qn, H)
    #Returns a propagator in H only evolving FermionicHilbertSpaces.indices corresponding to the specified qn
    index = FermionicHilbertSpaces.indices(qn, H)
    hamiltonian_sub = hamiltonian[index, index]
    propagator_sub = propagator(t, hamiltonian_sub)
    propagator_padded = spzeros(Complex{Float64}, dim(H), dim(H))
    propagator_padded[index, index] = propagator_sub
    return propagator_padded
end


operator_time_evolution(propagator, operator) = propagator' * operator * propagator
operator_time_evolution(operator, t, hamiltonian) = operator_time_evolution(propagator(t, hamiltonian), operator)
operator_time_evolution(operator, t, hamiltonian, qn, H) = operator_time_evolution(propagator(t, hamiltonian, qn, H), operator)


state_time_evolution(propagator, ρ) = propagator * ρ * propagator'
state_time_evolution(ρ, t, hamiltonian) = state_time_evolution(propagator(t, hamiltonian), ρ)
state_time_evolution(ρ, t, hamiltonian, H, qn) = state_time_evolution(propagator(t, hamiltonian, qn, H), ρ)
