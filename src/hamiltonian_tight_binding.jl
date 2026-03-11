struct DotParams
    ϵ::Any
    ϵb::Any
    u_intra::Any
end

struct InteractionParams
    t::Any
    t_so::Any
    u_inter::Any
end

struct Hamiltonians
    hamiltonian_main::Any
    hamiltonian_reservoir::Any
    hamiltonian_intersection::Any
    hamiltonian_total::Any
    dot_params_main::Any
    dot_params_reservoir::Any
    interaction_params::Any
end

## ============ Singel dot ================

function hamiltonian_ϵ(ϵ, u_intra, coordinate_labels)
    sum(
        (ϵ[label] - u_intra[label] / 2) * f[label, σ]' * f[label, σ]
        for σ in [:↑, :↓], label in coordinate_labels;
        init = 0
    )
end
function hamiltonian_b(ϵb, coordinate_labels)
    sum(
        #1/2 as normalization factor for pauli matrices
        1 / 2 * ϵb[label][1] * (f[label, :↑]'f[label, :↓] + f[label, :↓]'f[label, :↑]) + # Bx
        1 / 2 * ϵb[label][2] *
        (-im * f[label, :↑]'f[label, :↓] + im * f[label, :↓]'f[label, :↑]) + #By
        1 / 2 * ϵb[label][3] * (f[label, :↑]'f[label, :↑] - f[label, :↓]'f[label, :↓]) #Bz
        for label in coordinate_labels;
        init = 0
    )
end

function hamiltonian_c_intra(u_intra, coordinate_labels)
    sum(
        u_intra[label] * f[label, :↑]'f[label, :↓]'f[label, :↓]f[label, :↑]
        for label in coordinate_labels;
        init = 0
    )
end

## ============ Interactions ================

function hamiltonian_c_inter(u_inter, coordinate_labels)
    hamiltonian_c_inter_x(u_inter, coordinate_labels) +
    hamiltonian_c_inter_y(u_inter, coordinate_labels)
end

function hamiltonian_c_inter_x(u_inter, coordinate_labels)
    sum(
        u_inter[(i, j), (i + 1, j)] *
        f[(i, j), σ1]'f[(i + 1, j), σ2]'f[(i + 1, j), σ2]f[(i, j), σ1]
        for σ1 in [:↑, :↓], σ2 in [:↑, :↓], (i, j) in coordinate_labels
        if (i + 1, j) in coordinate_labels;
        init = 0
    )
end
function hamiltonian_c_inter_y(u_inter, coordinate_labels)
    sum(
        u_inter[(i, j), (i, j + 1)] *
        f[(i, j), σ1]'f[(i, j + 1), σ2]'f[(i, j + 1), σ2]f[(i, j), σ1]
        for σ1 in [:↑, :↓], σ2 in [:↑, :↓], (i, j) in coordinate_labels
        if (i, j + 1) in coordinate_labels;
        init = 0
    )
end

function hamiltonian_t(t, coordinate_labels)
    hamiltonian_t_x(t, coordinate_labels) + hamiltonian_t_y(t, coordinate_labels)
end

function hamiltonian_t_x(t, coordinate_labels)
    sum(
        t[(i, j), (i + 1, j)]f[(i + 1, j), σ]'f[(i, j), σ] + hc
        for σ in [:↑, :↓], (i, j) in coordinate_labels if (i + 1, j) in coordinate_labels;
        init = 0
    )
end
function hamiltonian_t_y(t, coordinate_labels)
    sum(
        t[(i, j), (i, j + 1)]f[(i, j + 1), σ]'f[(i, j), σ] + hc
        for σ in [:↑, :↓], (i, j) in coordinate_labels if (i, j + 1) in coordinate_labels;
        init = 0
    )
end

function hamiltonian_so(t_so, coordinate_labels)
    hamiltonian_so_x(t_so, coordinate_labels) +
    hamiltonian_so_y(t_so, coordinate_labels)
end

function hamiltonian_so_x(t_so, coordinate_labels)
    sum(
        t_so[(i, j), (i + 1, j)] *
        (-f[(i + 1, j), :↑]'f[(i, j), :↓] + f[(i + 1, j), :↓]'f[(i, j), :↑]) + hc
        for (i, j) in coordinate_labels if (i + 1, j) in coordinate_labels;
        init = 0
    )
end
function hamiltonian_so_y(t_so, coordinate_labels)
    sum(
        t_so[(i, j), (i, j + 1)] *
        (im * f[(i, j + 1), :↑]'f[(i, j), :↓] + im * f[(i, j + 1), :↓]'f[(i, j), :↑]) + hc
        for (i, j) in coordinate_labels if (i, j + 1) in coordinate_labels;
        init = 0
    )
end

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
    t_so = Dict(coupled_coordinate => t_so_func()
    for coupled_coordinate in coupled_coordinates)
    u_inter = Dict(coupled_coordinate => u_inter_func()
    for coupled_coordinate in coupled_coordinates)
    return InteractionParams(t, t_so, u_inter)
end

function get_coupled_coordinates(coordinates)
    coupled_coordinates_x = [((i, j), (i + 1, j))
                             for (i, j) in coordinates
                             if (i + 1, j) in coordinates]
    coupled_coordinates_y = [((i, j), (i, j + 1))
                             for (i, j) in coordinates
                             if (i, j + 1) in coordinates]
    return vcat(coupled_coordinates_x, coupled_coordinates_y)
end

function default_main_system_dot_params(coordinates)
    ϵ_func() = 0.5
    ϵb_func() = [0, 0, 1]
    u_intra_func() = rand() + 10
    return set_dot_params(ϵ_func, ϵb_func, u_intra_func, coordinates)
end

function default_reservoir_dot_params(coordinates)
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

function hamiltonian_dots(dot_params, coordinates)
    hamiltonian_ϵ(dot_params.ϵ, dot_params.u_intra, coordinates) +
    hamiltonian_b(dot_params.ϵb, coordinates) +
    hamiltonian_c_intra(dot_params.u_intra, coordinates)
end

function hamiltonian_interactions(interaction_params, coordinates)
    hamiltonian_t(interaction_params.t, coordinates) +
    hamiltonian_so(interaction_params.t_so, coordinates) +
    hamiltonian_c_inter(interaction_params.u_inter, coordinates)
end

function hamiltonian_interactions_x(interaction_params, coordinates)
    hamiltonian_t_x(interaction_params.t, coordinates) +
    hamiltonian_so_x(interaction_params.t_so, coordinates) +
    hamiltonian_c_inter_x(interaction_params.u_inter, coordinates)
end

function hamiltonians(qd_system, seed = nothing)
    isnothing(seed) || Random.seed!(seed)
    dot_params_main = default_main_system_dot_params(qd_system.grid.main)
    dot_params_reservoir = default_reservoir_dot_params(qd_system.grid.res)
    interaction_params = default_interaction_params(qd_system.grid.total)
    hamiltonians(qd_system, dot_params_main, dot_params_reservoir, interaction_params)
end

function hamiltonians_equal_param(qd_system)
    dot_params_main = default_equal_dot_params(qd_system.grid.main)
    dot_params_reservoir = default_equal_dot_params(qd_system.grid.res)
    interaction_params = default_equal_interaction_params(qd_system.grid.total)
    hamiltonians(qd_system, dot_params_main, dot_params_reservoir, interaction_params)
end

function hamiltonians(
        qd_system, dot_params_main::DotParams, dot_params_reservoir::DotParams,
        interaction_params::InteractionParams)
    grid = qd_system.grid
    hamiltonian_main = hamiltonian_dots(dot_params_main, grid.main) +
                       hamiltonian_interactions(interaction_params, grid.main)
    hamiltonian_reservoir = hamiltonian_dots(dot_params_reservoir, grid.res) +
                            hamiltonian_interactions(interaction_params, grid.res)
    hamiltonian_intersection = hamiltonian_interactions_x(
        interaction_params, grid.intersection)
    hamiltonian_total = hamiltonian_main + hamiltonian_reservoir + hamiltonian_intersection
    return Hamiltonians(hamiltonian_main, hamiltonian_reservoir,
        hamiltonian_intersection, hamiltonian_total,
        dot_params_main, dot_params_reservoir, interaction_params)
end

function matrix_representation_hams(hams::Hamiltonians, qd_system)
    Hamiltonians(
        hams.hamiltonian_main,
        matrix_representation(hams.hamiltonian_reservoir, qd_system.H_reservoir),
        hams.hamiltonian_intersection,
        matrix_representation(hams.hamiltonian_total, qd_system.H_total),
        hams.dot_params_main,
        hams.dot_params_reservoir,
        hams.interaction_params
    )
end