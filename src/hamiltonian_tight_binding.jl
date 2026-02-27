struct DotParams
    ϵ
    ϵb
    u_intra
end

struct InteractionParams
    t
    t_so
    u_inter
end

struct Hamiltonians
    hamiltonian_main
    hamiltonian_reservoir
    hamiltonian_intersection
    hamiltonian_total
    dot_params_main
    dot_params_reservoir
    interaction_params
end

## ============ Singel dot ================

hamiltonian_ϵ(ϵ, u_intra, coordinate_labels, f) = sum(
    (ϵ[label] - u_intra[label] / 2) * f[label, σ]' * f[label, σ]
    for σ ∈ [:↑, :↓], label ∈ coordinate_labels;
    init=0
)
hamiltonian_b(ϵb, coordinate_labels, f) = sum(
    #1/2 as normalization factor for pauli matrices
    1 / 2 * ϵb[label][1] * (f[label, :↑]'f[label, :↓] + f[label, :↓]'f[label, :↑]) + # Bx
    1 / 2 * ϵb[label][2] * (-im * f[label, :↑]'f[label, :↓] + im * f[label, :↓]'f[label, :↑]) + #By
    1 / 2 * ϵb[label][3] * (f[label, :↑]'f[label, :↑] - f[label, :↓]'f[label, :↓]) #Bz
    for label ∈ coordinate_labels;
    init=0
)

hamiltonian_c_intra(u_intra, coordinate_labels, f) = sum(
    u_intra[label] * f[label, :↑]'f[label, :↓]'f[label, :↓]f[label, :↑]
    for label ∈ coordinate_labels;
    init=0
)

## ============ Interactions ================

hamiltonian_c_inter(u_inter, coordinate_labels, f) = hamiltonian_c_inter_x(u_inter, coordinate_labels, f) + hamiltonian_c_inter_y(u_inter, coordinate_labels, f)

hamiltonian_c_inter_x(u_inter, coordinate_labels, f) = sum(
    u_inter[(i, j), (i + 1, j)] * f[(i, j), σ1]'f[(i + 1, j), σ2]'f[(i + 1, j), σ2]f[(i, j), σ1]
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], (i, j) ∈ coordinate_labels if (i + 1, j) ∈ coordinate_labels;
    init=0
)
hamiltonian_c_inter_y(u_inter, coordinate_labels, f) = sum(
    u_inter[(i, j), (i, j + 1)] * f[(i, j), σ1]'f[(i, j + 1), σ2]'f[(i, j + 1), σ2]f[(i, j), σ1]
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], (i, j) ∈ coordinate_labels if (i, j + 1) ∈ coordinate_labels;
    init=0
)

hamiltonian_t(t, coordinate_labels, f) = hamiltonian_t_x(t, coordinate_labels, f) + hamiltonian_t_y(t, coordinate_labels, f)

hamiltonian_t_x(t, coordinate_labels, f) = sum(
    t[(i, j), (i + 1, j)]f[(i + 1, j), σ]'f[(i, j), σ] + hc
    for σ ∈ [:↑, :↓], (i, j) ∈ coordinate_labels if (i + 1, j) ∈ coordinate_labels;
    init=0
)
hamiltonian_t_y(t, coordinate_labels, f) = sum(
    t[(i, j), (i, j + 1)]f[(i, j + 1), σ]'f[(i, j), σ] + hc
    for σ ∈ [:↑, :↓], (i, j) ∈ coordinate_labels if (i, j + 1) ∈ coordinate_labels;
    init=0
)

hamiltonian_so(t_so, coordinate_labels, f) = hamiltonian_so_x(t_so, coordinate_labels, f) + hamiltonian_so_y(t_so, coordinate_labels, f)

hamiltonian_so_x(t_so, coordinate_labels, f) = sum(
    t_so[(i, j), (i + 1, j)] * (-f[(i + 1, j), :↑]'f[(i, j), :↓] + f[(i + 1, j), :↓]'f[(i, j), :↑]) + hc
    for (i, j) ∈ coordinate_labels if (i + 1, j) ∈ coordinate_labels;
    init=0
)
hamiltonian_so_y(t_so, coordinate_labels, f) = sum(
    t_so[(i, j), (i, j + 1)] * (im * f[(i, j + 1), :↑]'f[(i, j), :↓] + im * f[(i, j + 1), :↓]'f[(i, j), :↑]) + hc
    for (i, j) ∈ coordinate_labels if (i, j + 1) ∈ coordinate_labels;
    init=0
)

## ========= Set Dot Parameters =============
function set_dot_params(ϵ_func, ϵb_func, u_intra_func, coordinates)
    ϵ = Dict(coordinate => ϵ_func() for coordinate in coordinates)
    ϵb = Dict(coordinate => ϵb_func() for coordinate in coordinates)
    u_intra = Dict(coordinate => u_intra_func() for coordinate in coordinates)
    return DotParams(ϵ, ϵb, u_intra)
end

function set_interaction_params(t_func, t_so_func, u_inter_func, coordinates)
    coupled_coordinates = get_coupled_coordinates(coordinates)
    t = Dict(coupled_coordinate => t_func() for coupled_coordinate in coupled_coordinates)
    t_so = Dict(coupled_coordinate => t_so_func() for coupled_coordinate in coupled_coordinates)
    u_inter = Dict(coupled_coordinate => u_inter_func() for coupled_coordinate in coupled_coordinates)
    return InteractionParams(t, t_so, u_inter)
end

function get_coupled_coordinates(coordinates)
    coupled_coordinates_x = [((i, j), (i + 1, j)) for (i, j) in coordinates
                             if (i + 1, j) ∈ coordinates]
    coupled_coordinates_y = [((i, j), (i, j + 1)) for (i, j) in coordinates
                             if (i, j + 1) ∈ coordinates]
    return vcat(coupled_coordinates_x, coupled_coordinates_y)
end

function default_main_system_dot_params(coordinates)
    ϵ_func() = 0.5
    ϵb_func() = [0, 0, 1]
    u_intra_func() = rand() + 10
    return set_dot_params(ϵ_func, ϵb_func, u_intra_func, coordinates)
end

function defalt_reservoir_dot_params(coordinates)
    ϵ_func() = rand()
    ϵb_func() = [0, 0, 1]
    u_intra_func() = rand() + 10
    return set_dot_params(ϵ_func, ϵb_func, u_intra_func, coordinates)
end


function default_interaction_params(coordinates)
    t_func() = rand()
    t_so_func() = 0.1 * rand()
    u_inter_func() = rand()
    return set_interaction_params(t_func, t_so_func, u_inter_func, coordinates)
end

function default_equal_dot_params(coordinates)
    ϵ_val() = 0.0
    ϵb_val() = [0, 0, 0]
    u_intra_val() = 10.0
    return set_dot_params(ϵ_val, ϵb_val, u_intra_val, coordinates)
end

function default_equal_interaction_params(coordinates)
    t_val() = 1.0
    t_so_val() = 0.0
    u_inter_val() = 0.0
    return set_interaction_params(t_val, t_so_val, u_inter_val, coordinates)
end

## ========= System Hamiltonians =============

hamiltonian_dots(dot_params, coordinates, f) =
    hamiltonian_ϵ(dot_params.ϵ, dot_params.u_intra, coordinates, f) + hamiltonian_b(dot_params.ϵb, coordinates, f) + hamiltonian_c_intra(dot_params.u_intra, coordinates, f)

hamiltonian_interactions(interaction_params, coordinates, f) =
    hamiltonian_t(interaction_params.t, coordinates, f) + hamiltonian_so(interaction_params.t_so, coordinates, f) + hamiltonian_c_inter(interaction_params.u_inter, coordinates, f)

hamiltonian_interactions_x(interaction_params, coordinates, f) =
    hamiltonian_t_x(interaction_params.t, coordinates, f) + hamiltonian_so_x(interaction_params.t_so, coordinates, f) + hamiltonian_c_inter_x(interaction_params.u_inter, coordinates, f)

function hamiltonians(quantum_dot_system, seed=nothing)
    isnothing(seed) || Random.seed!(seed)
    dot_params_main = default_main_system_dot_params(quantum_dot_system.coordinates_main)
    dot_params_reservoir = defalt_reservoir_dot_params(quantum_dot_system.coordinates_reservoir)
    interaction_params = default_interaction_params(quantum_dot_system.coordinates_total)
    hamiltonians(quantum_dot_system, dot_params_main, dot_params_reservoir, interaction_params)
end

function hamiltonians_equal_param(quantum_dot_system)
    dot_params_main = default_equal_dot_params(quantum_dot_system.coordinates_main)
    dot_params_reservoir = default_equal_dot_params(quantum_dot_system.coordinates_reservoir)
    interaction_params = default_equal_interaction_params(quantum_dot_system.coordinates_total)
    hamiltonians(quantum_dot_system, dot_params_main, dot_params_reservoir, interaction_params)
end

function hamiltonians(quantum_dot_system, dot_params_main::DotParams, dot_params_reservoir::DotParams, interaction_params::InteractionParams)
    hamiltonian_main = hamiltonian_dots(dot_params_main, quantum_dot_system.coordinates_main, quantum_dot_system.f) + hamiltonian_interactions(interaction_params, quantum_dot_system.coordinates_main, quantum_dot_system.f)
    hamiltonian_reservoir = hamiltonian_dots(dot_params_reservoir, quantum_dot_system.coordinates_reservoir, quantum_dot_system.f) + hamiltonian_interactions(interaction_params, quantum_dot_system.coordinates_reservoir, quantum_dot_system.f)
    hamiltonian_intersection = hamiltonian_interactions_x(interaction_params, quantum_dot_system.coordinates_intersection, quantum_dot_system.f)
    hamiltonian_total = hamiltonian_main + hamiltonian_reservoir + hamiltonian_intersection
    return Hamiltonians(hamiltonian_main, hamiltonian_reservoir, hamiltonian_intersection, hamiltonian_total,
        dot_params_main, dot_params_reservoir, interaction_params)
end
