## Standard time evolution
propagator(t, hamiltonian) = cis(-t * Matrix(hamiltonian))

operator_time_evolution(propagator, operator) = propagator' * operator * propagator
operator_time_evolution(operator, t, hamiltonian) = operator_time_evolution(propagator(t, hamiltonian), operator)

state_time_evolution(propagator, ρ) = propagator * ρ * propagator'
state_time_evolution(ρ, t, hamiltonian) = state_time_evolution(propagator(t, hamiltonian), ρ)

## Only evolve a qn-sector

propagator(t, hamiltonian, qn::Number, H::AbstractHilbertSpace) = propagator(t, hamiltonian, sector(qn, H), H)
function propagator(t, hamiltonian::AbstractMatrix{T}, H_qn::AbstractHilbertSpace, H ::AbstractHilbertSpace) where T
    index = FermionicHilbertSpaces.indices(H_qn, H)
    hamiltonian_sub = hamiltonian[index, index]
    propagator_sub = propagator(t, hamiltonian_sub)
    propagator_padded = spzeros(T, dim(H), dim(H))
    propagator_padded[index, index] = propagator_sub
    return propagator_padded
end

operator_time_evolution(operator, t, hamiltonian, qn::Number, H) = operator_time_evolution(propagator(t, hamiltonian, sector(qn, H), H), operator)
operator_time_evolution(operator, t, hamiltonian, H_qn::AbstractHilbertSpace, H) = operator_time_evolution(propagator(t, hamiltonian, H_qn, H), operator)

state_time_evolution(ρ, t, hamiltonian, qn::Number, H) = state_time_evolution(propagator(t, hamiltonian, sector(qn,H), H), ρ)
state_time_evolution(ρ, t, hamiltonian, H_qn::AbstractHilbertSpace, H) = state_time_evolution(propagator(t, hamiltonian, H_qn, H), ρ)