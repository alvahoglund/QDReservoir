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

const PauliKeys = (:σ0, :σx, :σy, :σz)
function paulis(H, Hfinal=H)
    dim(H) == 2 || throw(ArgumentError("Paulis is only defined for 2-dimensional Hilbert spaces"))
    coords = sites(H)
    x = only(coords)
    I = embed([1 0; 0 1], H => Hfinal)
    X = embed([0 1; 1 0], H => Hfinal)
    Y = embed([0 -im; im 0], H => Hfinal)
    Z = embed([1 0; 0 -1], H => Hfinal)
    vals = [I, X, Y, Z]
    return Dict(map(Pair, PauliKeys, vals))
end
function pauli_strings(Hs, Hfinal)
    ps = map(H -> collect(paulis(H, Hfinal)), Hs)
    pairs = map(Iterators.product(ps...)) do pairs
        key = tuple(first.(pairs)...)
        mat = prod(last.(pairs))
        key => mat
    end
    Dict(pairs)
end
function pauli_matrix(Hs, Hfinal)
    ps = pauli_strings(Hs, Hfinal)
    P = stack(vec, ps[a, b] for a in PauliKeys for b in PauliKeys)
    # each column of P is a vectorized Pauli string
    return transpose(P)
end

process_complex(value, tolerance=1e-3) = abs(imag(value)) < tolerance ? real(value) : throw(ArgumentError("The value has an imaginary part: $(imag(value))"))
expectation_value(ρ, op::AbstractMatrix) = process_complex((tr(density_matrix(ρ) * op)))
variance(ρ, op) = expectation_value(ρ, op^2) - expectation_value(ρ, op)^2

## ======== Measurement sets =================

function charge_measurements(qd_system)
    coordinates = qd_system.grid.total
    f = qd_system.f
    single_charge_sym_ops = [nbr_op(coordinate, f) for coordinate in coordinates]
    double_charge_sym_ops = [nbr2_op(coordinate, f) for coordinate in coordinates]
    symops = vcat(single_charge_sym_ops, double_charge_sym_ops)
    [matrix_representation(op, qd_system.H_total) for op in symops]
end

single_charge_probabilities(coordinates, f) = [p1(coordinate, f) for coordinate in coordinates]
double_charge_probabilities(coordinates, f) = [p2(coordinate, f) for coordinate in coordinates]
charge_probabilities(coordinates, f) = vcat(single_charge_probabilities(coordinates, f), double_charge_probabilities(coordinates, f))
function charge_probabilities(qd_system)
    coordinates = qd_system.grid.total
    f = qd_system.f
    charge_prob_ops = charge_probabilities(coordinates, f)
    [matrix_representation(op, qd_system.H_total) for op in charge_prob_ops]
end
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
correlated_measurements(qd_system) = correlated_measurements(qd_system.grid.total, qd_system.qn_total, qd_system.f)

## ============= Spin measurements ======================

#Operator for total spin S^2 on coodinate i 
Si2(coordinate_i, f, H_i) = matrix_representation(3 / 4 * p1(coordinate_i, f), H_i)

# Operator for S_i ⋅ S_j
function Sij(coordinate_i, coordinate_j, H)
    Hi = hilbert_space(labels((coordinate_i,)), NumberConservation(1))
    Hj = hilbert_space(labels((coordinate_j,)), NumberConservation(1))
    ps = pauli_strings((Hi, Hj), H)
    1 / 4 * sum(ps[σ, σ] for σ in [:σx, :σy, :σz])
end

# S^2 operator 
function total_spin_op(coordinates, f, H)
    S2_op = sum([Si2(coordinate, f, H) for coordinate in coordinates])
    Sij_op = sum((Sij(coordinate_i, coordinate_j, H)
                  for (i, coordinate_i) in enumerate(coordinates)
                  for (j, coordinate_j) in enumerate(coordinates)
                  if i < j), 
                    init = zero(S2_op))
    return S2_op + 2 * Sij_op
end

#S^2 = S(S+1) if the state is an eigenstate of S^2
s_from_s2(s2_val) = -1 / 2 + √(s2_val + 1 / 4)