## ============ Measurements =================
σ0(i, f) = (f[i, :↑]' * f[i, :↑] + f[i, :↓]' * f[i, :↓])
σx(i, f) = (f[i, :↓]' * f[i, :↑] + f[i, :↑]' * f[i, :↓])
σy(i, f) = im * (f[i, :↓]' * f[i, :↑] - f[i, :↑]' * f[i, :↓])
σz(i, f) = (f[i, :↑]' * f[i, :↑] - f[i, :↓]' * f[i, :↓])

nbr_op(coordinate, f) = f[coordinate, :↑]' * f[coordinate, :↑] + f[coordinate, :↓]' * f[coordinate, :↓]
nbr2_op(coordinate, f) = f[coordinate, :↑]' * f[coordinate, :↑] * f[coordinate, :↓]' * f[coordinate, :↓]

p(nbr_index, coordinate, f) = eval(Expr(:call, Symbol("p", nbr_index), coordinate, f))
p0(coordinate, f) = 1 - p1(coordinate, f) - p2(coordinate, f) #Probability to measure 0 charge
p1(coordinate, f) = nbr_op(coordinate, f) - 2 * nbr2_op(coordinate, f) # Probability to measure 1 charge
p2(coordinate, f) = nbr2_op(coordinate, f) # Probability to measure 2 charges

pauli_string(σi, σj, coordinate_i, coordinate_j, H_i, H_j, H_ij, f) = 
    tensor_product((matrix_representation(σi(coordinate_i, f), H_i),
                    matrix_representation(σj(coordinate_j, f), H_j)),
                    (H_i, H_j) => H_ij)

pauli_string_main(σi, σj, qd_system) = 
    pauli_string(σi, σj, qd_system.coordinates_main[1], qd_system.coordinates_main[2], qd_system.H_main_a_qn, qd_system.H_main_b_qn, qd_system.H_main_qn, qd_system.f)

process_complex(value, tolerance=1e-3) = abs(imag(value)) < tolerance ? real(value) : throw(ArgumentError("The value has an imaginary part: $(imag(value))"))
expectation_value(ρ, op) = process_complex((tr(ρ * op)))
variance(ρ, op) = expectation_value(ρ, op^2) - expectation_value(ρ, op)^2

## ======== Measurement sets =================

pauli_strings(coordinate_i, coordinate_j, H_i, H_j, H_ij, f) = 
    [pauli_string(σi, σj, coordinate_i, coordinate_j, H_i, H_j, H_ij, f) for σi in [σ0, σx, σy, σz] for σj in [σ0, σx, σy, σz]]
pauli_string_labels() = ["$(a) ⊗ $(b)" for a in ["σ0", "σx", "σy", "σz"] for b in ["σ0", "σx", "σy", "σz"]]

#A matrix where row is a row vectorized pauli matrix:  |σ_i ⊗ σ_j) = vec(σ_i ⊗ σ_j)^†
pauli_string_matrix(coordinate_i, coordinate_j, H_i, H_j, H_ij, f) = 
    vcat([reshape(pauli_string, dim(H_ij)^2, 1)' for pauli_string in pauli_strings(coordinate_i, coordinate_j, H_i, H_j, H_ij, f)]...)

pauli_string_matrix_main(qd_system) = 
    pauli_string_matrix(qd_system.coordinates_main[1], qd_system.coordinates_main[2], qd_system.H_main_a_qn, qd_system.H_main_b_qn, qd_system.H_main_qn, qd_system.f)

single_charge_measurements(coordinates, f) = [nbr_op(coordinate, f) for coordinate in coordinates]
double_charge_measurements(coordinates, f) = [nbr2_op(coordinate, f) for coordinate in coordinates]
charge_measurements(coordinates, f) = vcat(single_charge_measurements(coordinates, f), double_charge_measurements(coordinates, f))

single_charge_probabilities(coordinates, f) = [p1(coordinate, f) for coordinate in coordinates]
double_charge_probabilities(coordinates, f) = [p2(coordinate, f) for coordinate in coordinates]
charge_probabilities(coordinates, f) = vcat(single_charge_probabilities(coordinates, f), double_charge_probabilities(coordinates, f))

function correlated_measurements(coordinates, qn_total, f)
    valid_combos = get_measurement_combinations(coordinates, qn_total)
    measurement_ops = [measurement_combination_op(coordinates, f, measurement_combo)
                       for measurement_combo in valid_combos]
    return measurement_ops
end
function get_measurement_combinations(coordinates, qn_total)
    nbr_coordinates = length(coordinates)
    all_combos = Iterators.product(ntuple(_ -> 0:2, nbr_coordinates)...)
    valid_combos = [measurement_combo
                    for measurement_combo in all_combos
                    if qn_total == sum(measurement_combo)]
    return valid_combos
end
measurement_combination_op(coordinates, f, measurement_combo) = prod([p(measurement_combo[i], coord, f) for (i, coord) in enumerate(coordinates)])
correlated_measurements(qd_system) = correlated_measurements(qd_system.coordinates_total, qd_system.qn_total, qd_system.f)

## ============= Spin measurements ======================

#Operator for total spin S^2 on coodinate i 
Si2(coordinate_i, f, H_i) = matrix_representation(3 / 4 * p1(coordinate_i, f), H_i)

# Operator for S_i ⋅ S_j
function Sij(coordinate_i, coordinate_j, f, H)
    Hi = hilbert_space(labels([coordinate_i]), NumberConservation())
    Hj = hilbert_space(labels([coordinate_j]), NumberConservation())
    Hij = hilbert_space(labels([coordinate_i, coordinate_j]), NumberConservation())

    σxi = matrix_representation(σx(coordinate_i, f), Hi)
    σxj = matrix_representation(σx(coordinate_j, f), Hj)
    σxij = tensor_product((σxi, σxj), (Hi, Hj) => Hij)

    σyi = matrix_representation(σy(coordinate_i, f), Hi)
    σyj = matrix_representation(σy(coordinate_j, f), Hj)
    σyij = tensor_product((σyi, σyj), (Hi, Hj) => Hij)

    σzi = matrix_representation(σz(coordinate_i, f), Hi)
    σzj = matrix_representation(σz(coordinate_j, f), Hj)
    σzij = tensor_product((σzi, σzj), (Hi, Hj) => Hij)

    S_ij = 1 / 4 * (σxij + σyij + σzij)

    return embed(S_ij, Hij => H)
end

# S^2 operator 
function total_spin_op(coordinates, f, H)
    S2_op = sum([Si2(coordinate, f, H) for coordinate in coordinates])
    Sij_op = sum([Sij(coordinate_i, coordinate_j, f, H)
                  for (i, coordinate_i) in enumerate(coordinates)
                  for (j, coordinate_j) in enumerate(coordinates)
                  if i < j])
    return S2_op + 2 * Sij_op
end

#S^2 = S(S+1) if the state is an eigenstate of S^2
s_from_s2(s2_val) = -1 / 2 + √(s2_val + 1 / 4)