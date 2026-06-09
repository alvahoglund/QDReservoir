
# ======================= Plotting Weight Matrix =============================
function plot_heatmap_W_spin_basis!(ax, W, S, sys)
    Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
    W_spin_basis = (QDR.process_complex.(S * Pm))' * W .* (1 / 4)

    hm = heatmap!(
        ax, 1:4, 1:4, reshape(W_spin_basis, (4, 4)), colormap = :bam, colorrange = [-1, 1])

    ax.xticks = (1:4, ["σ0", "σx", "σy", "σz"])
    ax.yticks = (1:4, ["σ0", "σx", " σy", "σz"])
    return hm
end

function plot_heatmap_W_spin_basis(W_list, S, sys, σE_list)
    fig = Figure()
    for (i, W) in enumerate(W_list)
        ax = Axis(fig[1, i])
        hm = plot_heatmap_W_spin_basis!(ax, W, S, sys)
        ax.title = "σE = $(format_σE(σE_list[i]))"
        if i == length(W_list)
            Colorbar(fig[1, end + 1], hm, label = "Coefficient")
        end
    end
    return fig
end

function format_σE(σE)
    σE == 0 && return "0"
    exp = floor(Int, log10(σE))
    mantissa = σE / 10.0^exp
    mantissa ≈ 1 ? "10^$exp" : "$(round(mantissa, digits=2))×10^$exp"
end

# ==================== Plot in XX, YY, ZZ space ============================

function plot_nonlinear_db_spin_space(
        Ω_sub_sep, Ω_sub_ent, W, feature_transformation_alg; n_grid = 25)
    Pm_sub = get_sub_spin_basis(sys)
    grid = range(-1, 1, length = n_grid)

    function eval_point(xx, yy, zz)
        X_vec = QDR.process_complex.(S * (Pm_sub * [1.0, xx, yy, zz] ./ 4))
        X_poly = QDR.feature_transformation(
            reshape(X_vec, 1, :), feature_transformation_alg)
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
    plot_states_ent = 500
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
        state_list, W, S, σE = 0, feature_transformation_alg = QDR.IdentityFeatureTransformation())
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "p", ylabel = "Classifier output")
    frac_correct = []
    nbr_test_states = 1000
    for (i, state) in enumerate(state_list)
        p_range_sep = range(2 / 3, 1, length = nbr_test_states)
        p_range_ent = range(0, 2 / 3, length = nbr_test_states)
        Ω_sep = stack(vec(QDR.werner_state(state, p, sys.H_main)) for p in p_range_sep)
        Ω_ent = stack(vec(QDR.werner_state(state, p, sys.H_main)) for p in p_range_ent)
        X_sep = QDR.process_complex.((S * Ω_sep)')
        X_ent = QDR.process_complex.((S * Ω_ent)')
        X̃_sep = QDR.add_noise(X_sep, σE)
        X̃_ent = QDR.add_noise(X_ent, σE)
        X_sep_poly = QDR.feature_transformation(X̃_sep, feature_transformation_alg)
        X_ent_poly = QDR.feature_transformation(X̃_ent, feature_transformation_alg)
        Y_sep_pred = X_sep_poly * W
        Y_ent_pred = X_ent_poly * W
        scatter!(ax, p_range_sep, Y_sep_pred)
        scatter!(ax, p_range_ent, Y_ent_pred)
        vlines!(
            ax, [2 / 3], linestyle = :dash, color = :grey)
        hlines!(
            ax, [0], linestyle = :dash, color = :red)
        Y_test = vcat(ones(nbr_test_states), (-1) .* ones(nbr_test_states))
        Y_pred = vcat(Y_sep_pred, Y_ent_pred)

        push!(frac_correct, get_fraction_correct_classes(Y_test, Y_pred))
    end
    return fig, frac_correct
end

function plot_ew_mse(σE_list, mse_list)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "MSE",
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
