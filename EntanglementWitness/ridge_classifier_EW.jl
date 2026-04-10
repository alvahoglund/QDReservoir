##
using QDReservoir
using LinearAlgebra, Statistics, GLMakie, Distributions, Random
import QDReservoir as QDR

## ======================= Implement EWs =============================
function default_system()
    nbr_dots_res = 6
    qn_res = 3
    sys = tight_binding_system(2, nbr_dots_res, qn_res)
    hams = QDR.matrix_representation_hams(hamiltonians(sys.grids), sys)
    return sys, hams
end

function default_scrambling(sys, hams)
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

function get_sep_states(nbr_sep_states, rank, sys)
    stack(vec(density_matrix(QDR.random_separable_state(
              rank, sys.Hs_main, sys.H_main)))
    for i in 1:nbr_sep_states)
end

function get_ent_states(nbr_ent_states, sys, state_names)
    p_list = range(0, 1 / 3, length = nbr_ent_states ÷ length(state_names))
    mapreduce(state -> stack(vec(QDR.werner_state(state, p, sys.H_main)) for p in p_list),
        hcat, state_names)
end

get_charge_measurements(S, Ω) = QDR.process_complex.((S * Ω)')

function add_noise(σE, X)
    E = rand(Normal(0, σE), size(X))
    return X + E
end

function split_data(X̃_ent, X̃_sep)
    X̃ = vcat(X̃_sep, X̃_ent)
    Y = vcat(ones(size(X̃_sep, 1)), (-1) .* ones(size(X̃_ent, 1)))

    perm = randperm(size(X̃, 1))
    X̃ = X̃[perm, :]
    Y = Y[perm]

    nbr_train = size(X̃, 1) ÷ 2
    X̃_train, X̃_test = X̃[1:nbr_train, :], X̃[(nbr_train + 1):end, :]
    Y_train, Y_test = Y[1:nbr_train], Y[(nbr_train + 1):end]
    return X̃_train, X̃_test, Y_train, Y_test
end

function construct_EW(X_ent, X_sep, σE::Number, feature_transformation_func = identity)
    X̃_ent = add_noise(σE, X_ent)
    X̃_sep = add_noise(σE, X_sep)
    construct_EW(X̃_ent, X̃_sep, feature_transformation_func)
end

function construct_EW(X̃_ent, X̃_sep, feature_transformation_func = identity)
    X̃_train, X̃_test, Y_train, Y_test = split_data(
        X̃_ent, X̃_sep)
    X̃_train_poly = feature_transformation_func(X̃_train)
    X̃_test_poly = feature_transformation_func(X̃_test)
    W, Y_pred = QDR.ridge_regression(X̃_train_poly, Y_train, X̃_test_poly)
    return W, Y_pred, Y_test
end

function test_EW(X, σE, W, Y, feature_transformation_func = identity)
    X_noisy = add_noise(σE, X)
    X_poly = feature_transformation_func(X_noisy)
    Y_pred = X_poly * W
    return get_fraction_correct(Y_pred, Y)
end

function get_ew_mse(X_ent, X_sep, σE, feature_transformation_func = identity)
    W, Y_pred, Y_true = construct_EW(X_ent, X_sep, σE, feature_transformation_func)
    return mse(Y_true, Y_pred)
end

function get_ew_fraction_correct(X_ent, X_sep, σE, feature_transformation_func = identity)
    W, Y_pred, Y_true = construct_EW(X_ent, X_sep, σE, feature_transformation_func)
    return get_fraction_correct(Y_true, Y_pred)
end

# ======================= Plotting Weight Matrix =============================
function plot_heatmap_W_spin_basis(W, S, sys)
    Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
    W_spin_basis = (QDR.process_complex.(S * Pm))' * W .* (1 / 4)

    fig, ax, hm = heatmap(
        1:4, 1:4, reshape(W_spin_basis, (4, 4)), colormap = :bam, colorrange = [-1, 1])

    ax.xticks = (1:4, ["σ0", "σx", "σy", "σz"])
    ax.yticks = (1:4, ["σ0", "σx", " σy", "σz"])
    Colorbar(fig[:, end + 1], hm, label = "Coefficient")
    display(fig)
end

function format_σE(σE)
    σE == 0 && return "0"
    exp = floor(Int, log10(σE))
    mantissa = σE / 10.0^exp
    mantissa ≈ 1 ? "10^$exp" : "$(round(mantissa, digits=2))×10^$exp"
end

function plot_bars_W_spin_basis(W_list, S, sys, σE_list)
    Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
    fig = Figure()
    ax = Axis(fig[1, 1], xticklabelrotation = π / 4)
    pauli_string_labels = ["$a ⊗ $b"
                           for a in ["I", "X", "Y", "Z"] for b in ["I", "X", "Y", "Z"]]
    ax.xticks = (1:16, pauli_string_labels)

    n = length(W_list)
    colors = Makie.wong_colors()
    for (i, (W, σE)) in enumerate(zip(W_list, σE_list))
        W_spin_basis = (QDR.process_complex.(S * Pm))' * W .* (1 / 4)
        barplot!(ax, 1:16, W_spin_basis; dodge = fill(i, 16), n_dodge = n,
            color = colors[i], label = "σE = $(format_σE(σE))")
    end
    axislegend(ax, position = :lb)
    display(fig)
end

# ==================== Plot in XX, YY, ZZ space ============================

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

function project_on_sub_spin_basis(Ω_ent, Ω_sep, W)
    Pm_sub = get_sub_spin_basis(sys)
    Ω_sub_ent = QDR.process_complex.(Ω_ent' * Pm_sub)
    Ω_sub_sep = QDR.process_complex.(Ω_sep' * Pm_sub)
    W_sub_spin = (QDR.process_complex.(S * Pm_sub))' * W .* (1 / 4)
    return Ω_sub_ent, Ω_sub_sep, W_sub_spin
end

function project_on_sub_spin_basis(Ω_ent, Ω_sep)
    Pm_sub = get_sub_spin_basis(sys)
    Ω_sub_ent = QDR.process_complex.(Ω_ent' * Pm_sub)
    Ω_sub_sep = QDR.process_complex.(Ω_sep' * Pm_sub)
    return Ω_sub_ent, Ω_sub_sep
end

function plot_nonlinear_db_spin_space(
        Ω_sub_sep, Ω_sub_ent, W, feature_transformation; n_grid = 25)
    Pm_sub = get_sub_spin_basis(sys)
    grid = range(-1, 1, length = n_grid)

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

function plot_linear_db_spin_space(Ω_sep_spin, Ω_ent_spin, W_spin)
    fig = Figure()
    ax = Axis3(fig[1, 1], xlabel = "XX", ylabel = "YY", zlabel = "ZZ",
        title = "Ridge regression classification in XX, YY, ZZ space")
    plot_states_sep = 10
    plot_states_ent = 5000
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

# =================== Plot accuracy ============================
get_fraction_correct(Y_true, Y_pred) = mean((Y_true .> 0) .== (Y_pred .> 0))
mse(Y_true, Y_pred) = mean((Y_true - Y_pred) .^ 2)

function plot_test_vs_pred_ew(Y_test, Y_pred)
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

function test_werner_state(
        state_list, W, S, feature_transformation_func = identity)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "p", ylabel = "Classifier output")
    for (i, state) in enumerate(state_list)
        p_range_sep = range(2 / 3, 1, length = 100)
        p_range_ent = range(0, 2 / 3, length = 100)
        Ω_sep = stack(vec(QDR.werner_state(state, p, sys.H_main)) for p in p_range_sep)
        Ω_ent = stack(vec(QDR.werner_state(state, p, sys.H_main)) for p in p_range_ent)
        X_sep = QDR.process_complex.((S * Ω_sep)')
        X_ent = QDR.process_complex.((S * Ω_ent)')
        X_sep_poly = feature_transformation_func(X_sep)
        X_ent_poly = feature_transformation_func(X_ent)
        Y_sep_pred = X_sep_poly * W
        Y_ent_pred = X_ent_poly * W
        scatter!(ax, p_range_sep, Y_sep_pred)
        scatter!(ax, p_range_ent, Y_ent_pred)
        vlines!(
            ax, [2 / 3], linestyle = :dash, color = :grey)
        hlines!(
            ax, [0], linestyle = :dash, color = :red)
    end
    #axislegend(position = :lt)
    display(fig)
end

function plot_ew_mse(σE_list, mse_list)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "Mean Squared Error",
        title = "MSE of Ridge regression for entanglement witness",
        xscale = log10)
    lines!(ax, σE_list, mse_list, label = "MSE")
    axislegend(position = :lt)
    display(fig)
end

function plot_ew_fraction_correct(σE_list, fraction_correct_list, vlines_list = nothing)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "Fraction incorrect",
        title = "Classification accuracy of Ridge regression for entanglement witness",
        xscale = log10)
    lines!(ax, σE_list, 1 .- fraction_correct_list, label = "Fraction incorrect")
    if vlines_list !== nothing
        vlines!(ax, vlines_list, linestyle = :dash, color = :red, label = "Examples")
    end
    axislegend(position = :lt)
    display(fig)
end

function test_werner_state_against_noise(
        state_list, W_list, S, σE_list, feature_transformation_func = identity)
    nbr_test_states = 1000
    Y = (-1) .* ones(nbr_test_states)
    X_states = [get_charge_measurements(S, get_ent_states(nbr_test_states, sys, [state]))
                for state in state_list]
    EW_performance = [[test_EW(X_state, σ_E, W, Y,
                           feature_transformation_func)
                       for (σ_E, W) in zip(σE_list, W_list)] for X_state in X_states]
    return EW_performance
end

function test_separable_state_against_noise(
        W_list, S, σE_list, feature_transformation_func = identity)
    nbr_test_states = 1000
    Y = ones(nbr_test_states)
    Ω_list = [get_sep_states(nbr_test_states, rank, sys) for rank in 1:4]
    X_sep_list = [get_charge_measurements(S, Ω) for Ω in Ω_list]

    EW_performance = [[test_EW(X_sep, σ_E, W, Y,
                           feature_transformation_func)
                       for (σ_E, W) in zip(σE_list, W_list)] for X_sep in X_sep_list]
    return EW_performance
end

function plot_state_performance_against_noise(state_labels, EW_performance, σE_list)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "Fraction incorrect",
        title = "Classification accuracy of Entanglement Witness",
        xscale = log10)
    for (i, state_label) in enumerate(state_labels)
        lines!(ax, σE_list, 1 .- EW_performance[i], label = "State: $state_label")
    end
    axislegend(position = :lt)
    display(fig)
end
