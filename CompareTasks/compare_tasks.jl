
includet("../EntanglementWitness/ridge_classifier_EW.jl")
includet("../Purity/predict_purity.jl")
includet("../PredictSpin/predic_spin.jl")
using JLD2
##
BLAS.set_num_threads(1)
function linear_ew_states(sys, nbr_sep_states, nbr_ent_states)
    Ω_sep = get_prod_states(nbr_sep_states, sys)
    state_names = [QDR.singlet]
    Ω_ent = get_ent_states(nbr_ent_states, sys, state_names)
    return Ω_sep, Ω_ent
end

function nonlinear_ew_states(sys, nbr_sep_states, nbr_ent_states)
    Ω_sep = get_sep_states(nbr_sep_states, sys)
    states = [QDR.singlet, QDR.triplet_0, QDR.triplet_plus, QDR.triplet_minus]
    Ω_ent = get_ent_states(nbr_ent_states, sys, states)
    return Ω_sep, Ω_ent
end

function linear_ew_performance(S, Ω_sep, Ω_ent, σE)
    X_sep = get_charge_measurements(S, Ω_sep)
    X_ent = get_charge_measurements(S, Ω_ent)
    return 1 - get_ew_fraction_correct(X_ent, X_sep, σE)
end

function nonlinear_ew_performance(S, Ω_sep, Ω_ent, σE)
    X_sep = get_charge_measurements(S, Ω_sep)
    X_ent = get_charge_measurements(S, Ω_ent)
    return 1 - get_ew_fraction_correct(
        X_ent, X_sep, σE, QDR.degree_2_polynomial_feature_transformation)
end
function purity_prediction_mse(S, Ω, σE)
    X = get_charge_measurements(S, Ω)
    Y = get_purity(Ω)
    return mean(get_purity_mse(X, Y, σE))
end
function spin_prediction_mse(S, Ω, Pm, σE)
    return mean(get_mse(Ω, S, Pm, σE))
end
function set_ham(
        grids, ϵ_func_main, ϵ_func_res, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_params = QDR.set_dot_params(
        ϵ_func_main, ϵb_func, u_intra_func, grids.main)
    res_params = QDR.set_dot_params(
        ϵ_func_res, ϵb_func, u_intra_func, grids.res)
    interaction_params = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(
        grids, main_system_params, res_params, interaction_params)
end

function randomize_system()
    nbr_dots_res = rand(4:6)
    qn_res = rand(0:nbr_dots_res)
    sys = QDR.tight_binding_system(2, nbr_dots_res, qn_res)
    hams = hamiltonians(sys.grids)
    nbr_t = rand(2:4)
    t = [100 * rand() for _ in 1:nbr_t]
    return sys, hams, t
end

function get_scrambling_map(sys, hams, t)
    hams = QDR.matrix_representation_hams(hams, sys)
    ψ_ground = ground_state(hams.res)
    m_ops = QDR.matrix_representation_ops(
        QDR.charge_probabilities(sys.grids.total), sys.H_total)
    return QDR.scrambling_map(
        sys, m_ops, ψ_ground, hams.total, t, QDR.PureStateSteppingPropagatorAlg())
end

function get_performances(S, Ω_sep_linear, Ω_ent_linear, Ω_sep_nonlinear,
        Ω_ent_nonlinear, Ω_purity, Ω_spin, Pm, σE)
    linear_ew_results = linear_ew_performance(S, Ω_sep_linear, Ω_ent_linear, σE)
    nonlinear_ew_results = nonlinear_ew_performance(S, Ω_sep_nonlinear, Ω_ent_nonlinear, σE)
    purity_mse = purity_prediction_mse(S, Ω_purity, σE)
    spin_mse = spin_prediction_mse(S, Ω_spin, Pm, σE)
    return linear_ew_results, nonlinear_ew_results, purity_mse, spin_mse
end

function get_performances_matrix(S_list, Ω_sep_linear, Ω_ent_linear, Ω_sep_nonlinear,
        Ω_ent_nonlinear, Ω_purity, Ω_spin, Pm, σE_list)
    n_S = length(S_list)
    n_σE = length(σE_list)
    linear_ew_results = Matrix{Float64}(undef, n_S, n_σE)
    nonlinear_ew_results = Matrix{Float64}(undef, n_S, n_σE)
    purity_mse_results = Matrix{Float64}(undef, n_S, n_σE)
    spin_mse_results = Matrix{Float64}(undef, n_S, n_σE)
    Threads.@threads :dynamic for idx in CartesianIndices(linear_ew_results)
        i, j = Tuple(idx)
        S = S_list[i]
        σE = σE_list[j]
        linear_ew_results[i, j], nonlinear_ew_results[i, j], purity_mse_results[i, j], spin_mse_results[i, j] = get_performances(
            S, Ω_sep_linear, Ω_ent_linear, Ω_sep_nonlinear,
            Ω_ent_nonlinear, Ω_purity, Ω_spin, Pm, σE)
    end
    return linear_ew_results, nonlinear_ew_results, purity_mse_results, spin_mse_results
end

function format_σE(σE)
    σE == 0 && return "0"
    exp = floor(Int, log10(σE))
    mantissa = σE / 10.0^exp
    mantissa ≈ 1 ? "10^$exp" : "$(round(mantissa, digits=2))×10^$exp"
end

function log_edges(v)
    logv = log10.(v)
    mids = (logv[1:(end - 1)] .+ logv[2:end]) ./ 2
    pushfirst!(mids, 2 * logv[1] - mids[1])
    push!(mids, 2 * logv[end] - mids[end])
    return 10 .^ mids
end
function bin_by_ssv(ssv_list, data_matrix, n_bins)
    log_min, log_max = extrema(log10.(ssv_list))
    edges = 10 .^ range(log_min, log_max, length = n_bins + 1)
    n_cols = size(data_matrix, 2)
    med = fill(NaN, n_bins, n_cols)
    q25 = fill(NaN, n_bins, n_cols)
    q75 = fill(NaN, n_bins, n_cols)
    for i in 1:n_bins
        mask = (edges[i] .<= ssv_list .<= edges[i + 1])
        if any(mask)
            med[i, :] = vec(median(data_matrix[mask, :], dims = 1))
            q25[i, :] = vec(mapslices(
                x -> quantile(x, 0.25), data_matrix[mask, :], dims = 1))
            q75[i, :] = vec(mapslices(
                x -> quantile(x, 0.75), data_matrix[mask, :], dims = 1))
        end
    end
    return edges, med, q25, q75
end

function plot_heatmap(ssv_list, linear_ew_results, nonlinear_ew_results,
        purity_mse_list, spin_mse_list, σE_list; n_bins = 9)
    x_edges, linear_binned, _, _ = bin_by_ssv(ssv_list, linear_ew_results, n_bins)
    _, nonlinear_binned, _, _ = bin_by_ssv(ssv_list, nonlinear_ew_results, n_bins)
    _, purity_binned, _, _ = bin_by_ssv(ssv_list, purity_mse_list, n_bins)
    _, spin_binned, _, _ = bin_by_ssv(ssv_list, spin_mse_list, n_bins)
    y_edges = log_edges(Float64.(σE_list))
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        ["10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "1"])
    yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2], ["10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²"])
    fig = Figure(size = (800, 800))
    ax_linear = Axis(
        fig[2, 1], xlabel = "Smallest Singular Value", ylabel = "Noise level (σE)",
        title = "Linear EW - Fraction Incorrect", xscale = Makie.Symlog10(1e-8), yscale = Makie.Symlog10(1e-7),
        xticks = xticks, yticks = yticks)
    ax_nonlinear = Axis(
        fig[2, 2], xlabel = "Smallest Singular Value", ylabel = "Noise level (σE)",
        title = "Nonlinear EW - Fraction Incorrect", xscale = Makie.Symlog10(1e-8), yscale = Makie.Symlog10(1e-7),
        xticks = xticks, yticks = yticks)
    ax_purity = Axis(
        fig[1, 2], xlabel = "Smallest Singular Value", ylabel = "Noise level (σE)",
        title = "Purity Prediction - MSE", xscale = Makie.Symlog10(1e-8), yscale = Makie.Symlog10(1e-7), xticks = xticks,
        yticks = yticks)
    ax_spin = Axis(
        fig[1, 1], xlabel = "Smallest Singular Value", ylabel = "Noise level (σE)",
        title = "Spin Prediction - MSE", xscale = Makie.Symlog10(1e-8), yscale = Makie.Symlog10(1e-7), xticks = xticks,
        yticks = yticks)
    hm11 = heatmap!(ax_linear, x_edges, y_edges, linear_binned)
    hm12 = heatmap!(ax_nonlinear, x_edges, y_edges, nonlinear_binned)
    hm21 = heatmap!(ax_purity, x_edges, y_edges, purity_binned)
    hm22 = heatmap!(ax_spin, x_edges, y_edges, spin_binned)
    Colorbar(fig[1, 3], hm11)
    Colorbar(fig[1, 4], hm12)
    Colorbar(fig[2, 3], hm21)
    Colorbar(fig[2, 4], hm22)
    display(fig)
end

function plot_lines(ssv_list, linear_ew_results, nonlinear_ew_results,
        purity_mse_list, spin_mse_list, σE_list; n_bins = 25, yscale = identity, include_bands = true)
    x_edges, linear_med, linear_q25, linear_q75 = bin_by_ssv(
        ssv_list, linear_ew_results, n_bins)
    _, nonlinear_med, nonlinear_q25, nonlinear_q75 = bin_by_ssv(
        ssv_list, nonlinear_ew_results, n_bins)
    _, purity_med, purity_q25, purity_q75 = bin_by_ssv(ssv_list, purity_mse_list, n_bins)
    _, spin_med, spin_q25, spin_q75 = bin_by_ssv(ssv_list, spin_mse_list, n_bins)

    x_centers = sqrt.(x_edges[1:(end - 1)] .* x_edges[2:end])
    colors = Makie.wong_colors()

    fig = Figure(size = (900, 800))
    titles = ["Spin Prediction - MSE", "Purity Prediction - MSE",
        "Linear EW - Fraction Incorrect", "Nonlinear EW - Fraction Incorrect"]
    ylabels = ["MSE", "MSE", "Fraction Incorrect", "Fraction Incorrect"]
    med_list = [spin_med, purity_med, linear_med, nonlinear_med]
    q25_list = [spin_q25, purity_q25, linear_q25, nonlinear_q25]
    q75_list = [spin_q75, purity_q75, linear_q75, nonlinear_q75]
    for (k, (title, ylabel, med, q25, q75)) in enumerate(zip(
        titles, ylabels, med_list, q25_list, q75_list))
        row = (k - 1) ÷ 2 + 1
        col = (k - 1) % 2 + 1
        ax = Axis(
            fig[row, col]; title = title, xlabel = "SSV", ylabel = ylabel,
            xscale = log10, yscale = yscale)
        for j in eachindex(σE_list)
            valid = .!isnan.(med[:, j])
            c = colors[mod1(j, length(colors))]
            if include_bands
                band!(ax, x_centers[valid], q25[valid, j], q75[valid, j];
                    color = (c, 0.2))
            end
            lines!(ax, x_centers[valid], med[valid, j];
                color = c, label = format_σE(σE_list[j]))
            scatter!(ax, x_centers[valid], med[valid, j]; color = c, markersize = 6)
        end
        axislegend(ax, "σE", position = :rt, nbanks = 1)
    end
    display(fig)
end

function plot_performance_vs_ssv(ssv_list, linear_ew_results, nonlinear_ew_results,
        purity_mse_list, spin_mse_list, σE_list)
    fig = Figure(size = (800, 800))

    # Two subplots: one for MSE, one for fraction incorrectly classified, One plot for each noise level
    for (i, σE) in enumerate(σE_list)
        ax1 = Axis(
            fig[i, 1], xlabel = "Smallest Singular Value", ylabel = "MSE",
            xscale = log10, title = "Noise level: σE = $(format_σE(σE))")
        ax2 = Axis(
            fig[i, 2], xlabel = "Smallest Singular Value",
            ylabel = "Fraction Incorrectly\n Classified", xscale = log10, title = "Noise level: σE = $(format_σE(σE))")
        lines!(ax2, ssv_list, linear_ew_results[:, i], label = "Linear EW")
        lines!(ax2, ssv_list, nonlinear_ew_results[:, i], label = "Nonlinear EW")
        lines!(ax1, ssv_list, purity_mse_list[:, i] ./ maximum(purity_mse_list[:, i]),
            label = "Purity Prediction")
        lines!(
            ax1, ssv_list, spin_mse_list[:, i] ./ maximum(spin_mse_list[:, i]),
            label = "Spin Prediction")
        axislegend(ax1, position = :rt)
        axislegend(ax2, position = :rt)
    end
    display(fig)
end

## ============== Generate states ================
nbr_sep_states = 10^4
nbr_ent_states = 10^4
nbr_train = (nbr_sep_states + nbr_ent_states) ÷ 2

Ω_sep_linear, Ω_ent_linear = linear_ew_states(sys, nbr_sep_states, nbr_ent_states)
Ω_sep_nonlinear, Ω_ent_nonlinear = nonlinear_ew_states(
    sys, nbr_sep_states, nbr_ent_states)
Ω_purity = random_mixed_states(nbr_sep_states + nbr_ent_states, sys)
Ω_spin = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main))
for i in 1:(nbr_sep_states + nbr_ent_states))

## ============== Performance of different tasks ================
sys, hams = default_system()
S = default_scrambling(sys, hams)
σE = 0.1
Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)

linear_ew_result = linear_ew_performance(S, Ω_sep_linear, Ω_ent_linear, σE)
nonlinear_ew_result = nonlinear_ew_performance(
    S, Ω_sep_nonlinear, Ω_ent_nonlinear, σE)
purity_mse = purity_prediction_mse(S, Ω_purity, σE)
spin_mse = spin_prediction_mse(S, Ω_spin, Pm, σE)

## ============== Performance for varying noise levels ================
sys, hams = default_system()
S = default_scrambling(sys, hams)
σE_list = 10 .^ range(-10, 0, length = 50)
linear_ew_results = vcat([linear_ew_performance(S, Ω_sep_linear, Ω_ent_linear, σE)
                          for σE in σE_list]...)
nonlinear_ew_results = vcat([nonlinear_ew_performance(
                                 S, Ω_sep_nonlinear, Ω_ent_nonlinear, σE) for σE in σE_list]...)
purity_mse_list = vcat([purity_prediction_mse(S, Ω_purity, σE) for σE in σE_list]...)
spin_mse_list = vcat([spin_prediction_mse(S, Ω_spin, Pm, σE) for σE in σE_list]...)
svdvals_S = svdvals(S)
fig = Figure(size = (700, 400))
ax1 = Axis(
    fig[1, 1], xlabel = "Noise level (σE)", ylabel = "Fraction Incorrectly Classified/MSE",
    title = "Performance of different tasks", xscale = log10)
lines!(ax1, σE_list, linear_ew_results, label = "Linear EW")
lines!(ax1, σE_list, nonlinear_ew_results, label = "Nonlinear EW")
lines!(ax1, σE_list, purity_mse_list ./ maximum(purity_mse_list),
    label = "Purity Prediction MSE")
lines!(ax1, σE_list, spin_mse_list ./ maximum(spin_mse_list),
    label = "Average Spin Prediction MSE")
vlines!(ax1, svdvals_S, linestyle = :dash, color = :grey,
    label = "σS")
axislegend(position = :lt)
display(fig)

## ============== Performance for varying reservoirs ================
nbr_S = 100
sys_list = [randomize_system() for _ in 1:nbr_S]
S_list = Vector{Any}(undef, length(sys_list))
Threads.@threads for i in eachindex(sys_list)
    S_list[i] = get_scrambling_map(sys_list[i]...)
end
ssv_list = [minimum(svdvals(S)) for S in S_list]
# Sort reservoirs by their smallest singular value
sorted_indices = sortperm(ssv_list)
ssv_list = ssv_list[sorted_indices]
S_list = S_list[sorted_indices]
sys_list = sys_list[sorted_indices]

σE_list = [10^-6, 10^-5, 10^-4, 10^-3, 10^-2]
linear_ew_results, nonlinear_ew_results, purity_mse_list, spin_mse_list = get_performances_matrix(
    S_list, Ω_sep_linear, Ω_ent_linear, Ω_sep_nonlinear,
    Ω_ent_nonlinear, Ω_purity, Ω_spin, Pm, σE_list)
#jldsave("EntanglementWitness\\compare_tasks_data\\compare_tasks_data2.jld2";
#    linear_ew_results, nonlinear_ew_results, purity_mse_list, spin_mse_list, ssv_list, σE_list)

## ============== Plotting ================
linear_ew_results, nonlinear_ew_results, purity_mse_list, spin_mse_list, ssv_list, σE_list = jldopen(
    "CompareTasks\\compare_tasks_data\\compare_tasks_data2.jld2", "r") do file
    linear_ew_results = read(file, "linear_ew_results")
    nonlinear_ew_results = read(file, "nonlinear_ew_results")
    purity_mse_list = read(file, "purity_mse_list")
    spin_mse_list = read(file, "spin_mse_list")
    ssv_list = read(file, "ssv_list")
    σE_list = read(file, "σE_list")
    return linear_ew_results,
    nonlinear_ew_results, purity_mse_list, spin_mse_list, ssv_list, σE_list
end
plot_performance_vs_ssv(ssv_list, linear_ew_results, nonlinear_ew_results,
    purity_mse_list, spin_mse_list, σE_list)
plot_lines(ssv_list, linear_ew_results, nonlinear_ew_results,
    purity_mse_list, spin_mse_list, σE_list; n_bins = 10,
    yscale = Makie.Symlog10(1e-3), include_bands = true)
plot_heatmap(ssv_list, linear_ew_results, nonlinear_ew_results,
    purity_mse_list, spin_mse_list, σE_list; n_bins = 25)
