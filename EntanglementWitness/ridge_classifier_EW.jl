##
using QDReservoir
using LinearAlgebra, Statistics, GLMakie, Distributions, Random
import QDReservoir as QDR

## ======================= Plotting Functions =============================
function heatmap_W_spin_basis(W, sys)
    Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
    W_spin_basis = (QDR.process_complex.(S * Pm))' * W .* (1 / 4)
    W_spin_basis

    fig, ax, hm = heatmap(
        1:4, 1:4, reshape(W_spin_basis, (4, 4)), colormap = :bam, colorrange = [-1, 1])

    ax.xticks = (1:4, ["σ0", "σx", "σy", "σz"])
    ax.yticks = (1:4, ["σ0", "σx", " σy", "σz"])
    Colorbar(fig[:, end + 1], hm, label = "Coefficient")
    display(fig)
end

function get_linear_db(W_spin)
    x_range = range(-1, 1, length = 100)
    y_range = range(-1, 1, length = 100)
    b = W_spin[1]
    wx = W_spin[2]
    wy = W_spin[3]
    wz = W_spin[4]
    db_plane = [-(b + wx * x + wy * y) / wz for x in x_range, y in y_range]
    return x_range, y_range, db_plane
end

function get_sub_spin_basis(sys)
    Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
    Pm[:,
        [Pm_dict[(:σ0, :σ0)], Pm_dict[(:σx, :σx)],
            Pm_dict[(:σy, :σy)], Pm_dict[(:σz, :σz)]]]
end

function project_on_3D_spin_space(Ω_ent, Ω_sep, W)
    Pm_sub = get_sub_spin_basis(sys)
    Ω_sub_ent = QDR.process_complex.(Ω_ent' * Pm_sub)
    Ω_sub_sep = QDR.process_complex.(Ω_sep' * Pm_sub)
    W_sub_spin = (QDR.process_complex.(S * Pm_sub))' * W .* (1 / 4)
    return Ω_sub_ent, Ω_sub_sep, W_sub_spin
end

function project_on_3D_spin_space(Ω_ent, Ω_sep)
    Pm_sub = get_sub_spin_basis(sys)
    Ω_sub_ent = QDR.process_complex.(Ω_ent' * Pm_sub)
    Ω_sub_sep = QDR.process_complex.(Ω_sep' * Pm_sub)
    return Ω_sub_ent, Ω_sub_sep
end

function plot_nonlinear_db_in_spin_space(
        Ω_sub_sep, Ω_sub_ent, W, feature_transformation; n_grid = 25)
    Pm_sub = get_sub_spin_basis(sys)
    grid = range(-1, 1, length = n_grid)

    # Evaluate the full nonlinear pipeline (S → process_complex → tanh → W) on a 3D grid
    # in (XX, YY, ZZ) spin space. Each grid point is reconstructed as:
    #   ρ = (1/4)(σ0⊗σ0 + xx·σx⊗σx + yy·σy⊗σy + zz·σz⊗σz)
    function eval_point(xx, yy, zz)
        X_vec = QDR.process_complex.(S * (Pm_sub * [1.0, xx, yy, zz] ./ 4))
        X_poly = feature_transformation(reshape(X_vec, 1, :))
        Float32(dot(vec(X_poly), W))
    end
    vals = Float32[eval_point(xx, yy, zz) for xx in grid, yy in grid, zz in grid]

    fig = Figure()
    ax = Axis3(fig[1, 1], xlabel = "XX", ylabel = "YY", zlabel = "ZZ",
        title = "Nonlinear decision boundary in spin space")

    scatter!(ax, Ω_sub_sep[1:10:end, 2], Ω_sub_sep[1:10:end, 3], Ω_sub_sep[1:10:end, 4],
        label = "Separable", markersize = 5)
    scatter!(
        ax, Ω_sub_ent[1:1000:end, 2], Ω_sub_ent[1:1000:end, 3], Ω_sub_ent[1:1000:end, 4],
        label = "Entangled", markersize = 5)

    # Isosurface at classifier output = 0 (the decision boundary)
    volume!(ax, (-1, 1), (-1, 1), (-1, 1), vals;
        algorithm = :iso, isovalue = 0.0f0, isorange = 0.1f0, alpha = 0.5)
    display(fig)
end

function plot_3D_spin_space(Ω_sep_spin, Ω_ent_spin, W_spin)
    fig = Figure()
    ax = Axis3(fig[1, 1], xlabel = "XX", ylabel = "YY", zlabel = "ZZ",
        title = "Ridge regression classification in XX, YY, ZZ space")
    plot_states_sep = 10
    plot_states_ent = 1000
    scatter!(ax,
        Ω_sep_spin[1:plot_states_sep:end, 2],
        Ω_sep_spin[1:plot_states_sep:end, 3],
        Ω_sep_spin[1:plot_states_sep:end, 4],
        label = "Separable states", markersize = 5
    )

    scatter!(ax,
        Ω_ent_spin[1:plot_states_ent:end, 2],
        Ω_ent_spin[1:plot_states_ent:end, 3],
        Ω_ent_spin[1:plot_states_ent:end, 4],
        label = "Entangled states", markersize = 5
    )
    x_range, y_range, db_plane = get_linear_db(W_spin)
    surface!(ax, x_range, y_range, db_plane, label = "Decision boundary")

    display(fig)
end
function plot_test_vs_pred(Y_test, Y_pred)
    #Sort first on Y_test, then on Y_pred
    sort_indices = sortperm(Y_pred)
    Y_test = Y_test[sort_indices]
    Y_pred = Y_pred[sort_indices]
    x_range = range(-1, 1, length = length(Y_pred))
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Sorted index", ylabel = "Label value",
        title = "Ridge regression for entanglement witness, fraction correct = $(round(get_fraction_correct(Y_test, Y_pred), digits = 4))")
    scatter!(ax, x_range, tanh.(Y_pred * 3), label = "Predicted labels (tanh scaled)")
    scatter!(ax, x_range, Y_test, label = "True labels")
    scatter!(ax, x_range, Y_pred, label = "Predicted labels")
    axislegend(position = :lt)
    display(fig)
end
get_fraction_correct(Y_true, Y_pred) = mean((Y_true .> 0) .== (Y_pred .> 0))

function get_ham(grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.main)
    reservoir_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.res)
    interaction_parameters = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(grids, main_system_parameters,
        reservoir_parameters, interaction_parameters)
end

function default_system()
    ϵ_func() = 0.5
    ϵb_func() = [0, 0, 1]
    u_intra_func() = rand() + 10
    t_func() = rand()
    t_so_func() = 0.1 * rand()

    nbr_dots_res = 6
    qn_res = 3
    return tight_binding_system(2, nbr_dots_res, qn_res)
end

function default_scrambling(sys)
    t = 100
    measurements = map(m -> matrix_representation(m, sys.H_total),
        QDR.charge_probabilities(sys.grids.total))
    return scrambling_map(sys, measurements, ground_state(hams.res),
        hams.total, t)
end

function get_prod_states(nbr_sep_states, sys)
    stack(vec(density_matrix(QDR.random_product_state(sys.Hs_main, sys.H_main)))
    for i in 1:nbr_sep_states)
end

function get_sep_states(nbr_sep_states, sys)
    stack(vec(density_matrix(QDR.random_separable_state(
              rand(1:3), sys.Hs_main, sys.H_main)))
    for i in 1:nbr_sep_states)
end

function get_ent_states(nbr_ent_states, sys, state_names)
    p_list = range(0, 1 / 3, length = nbr_ent_states ÷ length(state_names))
    mapreduce(state -> stack(vec(QDR.werner_state(state, p, sys.H_main)) for p in p_list),
        hcat, state_names)
end

function get_charge_measurements(S, Ω, σE)
    X = QDR.process_complex.((S * Ω)')
    E = rand(Normal(0, σE), size(X))
    X̃ = X + E
    return X, X̃
end

function split_data(X_ent, X̃_ent, X_sep, X̃_sep)
    X = vcat(X_sep, X_ent)
    X̃ = vcat(X̃_sep, X̃_ent)

    # Shuffle data in X
    perm = randperm(size(X, 1))
    X = X[perm, :]
    X̃ = X̃[perm, :]
    nbr_train = size(X, 1) ÷ 2
    X_train, X_test = X[1:nbr_train, :], X[(nbr_train + 1):end, :]
    X̃_train, X̃_test = X̃[1:nbr_train, :], X̃[(nbr_train + 1):end, :]
    Y = vcat((-1) .* ones(size(X_ent)[1]), ones(size(X_ent)[1]))[perm]
    Y_train, Y_test = Y[1:nbr_train], Y[(nbr_train + 1):end]
    return X_train, X_test, X̃_train, X̃_test, Y_train, Y_test
end

function ridge_regression(X_train, Y_train, X_test, λ)
    W = (X_train' * X_train + λ * I) \ (X_train' * Y_train)
    Y_pred = X_test * W
    return W, Y_pred
end

# Degree-2 polynomial: [x, x², xᵢxⱼ]
# Equivalent to the degree-2 polynomial kernel k(x,y) = (xᵀy)²
function explicit_degree_2_polynomial_feature_transformation(X)
    n_samples, n_features = size(X)
    hcat(X, X .^ 2,
        [X[:, i] .* X[:, j] for i in 1:n_features for j in (i + 1):n_features]...)
end

# Random Fourier Features: approximates the RBF (Gaussian) kernel k(x,y) = exp(-||x-y||²/2σ²)
# ω is drawn once and reused — make sure to use the same ω for train and test.
# Returns a closure so the same random projection is applied consistently.
function make_rff_transformation(n_input_features; n_rff = 500, σ = 1.0)
    ω = randn(n_input_features, n_rff) ./ σ
    b = rand(Uniform(0, 2π), n_rff)
    X -> sqrt(2 / n_rff) .* cos.(X * ω .+ b')
end

## ============ LINEAR ENTANGLEMENT WITNESS FOR SINGLET STATE =====================
sys = default_system()
S = default_scrambling(sys)

σE = 10^-4
λ = 0

# Generate data
nbr_sep_states = 10^5
nbr_ent_states = 10^5
nbr_train = (nbr_sep_states + nbr_ent_states) ÷ 2
Ω_sep = get_prod_states(nbr_sep_states, sys)
state_names = [QDR.singlet]
Ω_ent = get_ent_states(nbr_ent_states, sys, state_names)
Ω = vcat(Ω_sep, Ω_ent)

X_ent, X̃_ent = get_charge_measurements(S, Ω_ent, σE)
X_sep, X̃_sep = get_charge_measurements(S, Ω_sep, σE)
X_train, X_test, X̃_train, X̃_test, Y_train, Y_test = split_data(
    X_ent, X̃_ent, X_sep, X̃_sep)
W, Y_pred = ridge_regression(X̃_train, Y_train, X̃_test, λ)

## Plots
heatmap_W_spin_basis(W, sys)
plot_test_vs_pred(Y_test, Y_pred)

Ω_sub_ent, Ω_sub_sep, W_sub_spin = project_on_3D_spin_space(Ω_ent, Ω_sep, W)
plot_3D_spin_space(Ω_sub_sep, Ω_sub_ent, W_sub_spin)

Ω_ent_noisy = (X̃_ent * pinv(S'))'
Ω_sep_noisy = (X̃_sep * pinv(S'))'
Ω_sub_ent_noisy, Ω_sub_sep_noisy, W_sub_spin = project_on_3D_spin_space(
    Ω_ent_noisy, Ω_sep_noisy, W)
plot_3D_spin_space(Ω_sub_sep_noisy, Ω_sub_ent_noisy, W_sub_spin)

## ============= NONLINEAR ENTANGLEMENT WITNESS FOR WERNER STATES    =====================
sys = default_system()
S = default_scrambling(sys)

σE = 10^-3
λ = 0

# Generate data
nbr_sep_states = 10^5
nbr_ent_states = 10^5
nbr_train = (nbr_sep_states + nbr_ent_states) ÷ 2
Ω_sep = get_sep_states(nbr_sep_states, sys)
state_names = [QDR.singlet, QDR.triplet_0, QDR.triplet_plus, QDR.triplet_minus]
Ω_ent = get_ent_states(nbr_ent_states, sys, state_names)
Ω = vcat(Ω_sep, Ω_ent)

X_ent, X̃_ent = get_charge_measurements(S, Ω_ent, σE)
X_sep, X̃_sep = get_charge_measurements(S, Ω_sep, σE)
X_train, X_test, X̃_train, X̃_test, Y_train, Y_test = split_data(
    X_ent, X̃_ent, X_sep, X̃_sep)

#feature_transformation_func = make_rff_transformation(size(X̃_train, 2))
feature_transformation_func = explicit_degree_2_polynomial_feature_transformation
X̃_train_poly = feature_transformation_func(X̃_train)
X̃_test_poly = feature_transformation_func(X̃_test)

W, Y_pred = ridge_regression(X̃_train_poly, Y_train, X̃_test_poly, λ)

plot_test_vs_pred(Y_test, Y_pred)
##

Ω_sub_ent, Ω_sub_sep = project_on_3D_spin_space(Ω_ent, Ω_sep)
plot_nonlinear_db_in_spin_space(
    Ω_sub_sep, Ω_sub_ent, W, feature_transformation_func)