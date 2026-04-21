using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions
import QDReservoir as QDR
##
function split_train_test(X, Y)
    nbr_train = size(X)[1] ÷ 2
    X_train, X_test = X[1:nbr_train, :], X[(nbr_train + 1):end, :]
    Y_train, Y_test = Y[1:nbr_train, :], Y[(nbr_train + 1):end, :]
    return X_train, X_test, Y_train, Y_test
end

function get_mse(Ω, S, Pm, σE)
    X = QDR.process_complex.((S * Ω)')
    E = rand(Normal(0, σE), size(X))
    X̃ = X + E

    Y = QDR.process_complex.((Pm' * Ω)')

    X_train, X_test, Y_train, Y_test = split_train_test(X̃, Y)

    W = pinv(X_train) * Y_train

    Y_pred = X_test * W
    return mean((Y_test - Y_pred) .^ 2, dims = 1)
end

function mse_prediction(S_SVD, Pm, σE, b)
    real.(diag(Pm' * S_SVD.V * diagm((b * σE^2) ./ (b .* S_SVD.S .^ 2 .+ σE^2)) * S_SVD.V' *
               Pm))
end

SV_overlap(S_SVD, Pm) = abs2.(S_SVD.V' * Pm)

function plot_varying_noise(Ω, S, Pm, Pm_dict, ps_list)
    S_SVD = svd(S)

    σE_list = 10 .^ range(-7, 0, length = 30)
    mse_list = vcat([get_mse(Ω, S, Pm, σE) for σE in σE_list]...)

    b = 0.0147
    sv_overlaps = SV_overlap(S_SVD, Pm)
    mse_pred_list = vcat([transpose(mse_prediction(S_SVD, Pm, σE, b)) for σE in σE_list]...)

    fig = Figure(size = (700, 400))
    for (i, ps) in enumerate(ps_list)
        idx = Pm_dict[ps...]
        ax = Axis(fig[i, 1], xlabel = "Noise level (σE)", ylabel = "Mean Squared Error",
            title = "Measurement: $(ps[1]) ⊗ $(ps[2])",
            xscale = log10)
        lines!(ax, σE_list, mse_list[:, idx], label = "MSE")
        #vlines!(ax, [10^-4, 10^-2], linestyle = :dash, color = :grey)
        lines!(ax, σE_list, mse_pred_list[:, idx], label = "Predicted MSE")
        vlines!(ax, sqrt(b) .* S_SVD.S, color = sv_overlaps[:, idx],
            colormap = :Blues, colorrange = (-0.5, maximum(sv_overlaps[:, idx])),
            linestyle = :dash, label = "√b*σS")
        axislegend(position = :lt)
    end
    display(fig)
end

function plot_mode_decomposition(mse_mat_small, mse_mat_large, S_SVD, Pm, Pm_dict, ps, b)
    idx = Pm_dict[ps...]
    sv_overlaps = SV_overlap(S_SVD, Pm)[:, idx]

    contributions = hcat([(b * σE^2) ./ (b .* S_SVD.S .^ 2 .+ σE^2) .* sv_overlaps
                          for σE in σE_list]...)'

    order = sortperm(S_SVD.S)
    contribs_sorted = contributions[:, order]
    σ_sorted = S_SVD.S[order]

    fig = Figure(size = (700, 350))
    ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "MSE contribution",
        title = "MSE of predicting $(ps[1]) ⊗ $(ps[2])", xscale = log10)

    σ_min, σ_max = extrema(σ_sorted)
    cmap = :viridis
    log_positions = (log10.(σ_sorted) .- log10(σ_min)) ./ (log10(σ_max) - log10(σ_min))
    colors = [Makie.interpolated_getindex(to_colormap(cmap), t) for t in log_positions]
    cumulative = zeros(length(σE_list))
    total_mse_max = maximum(sum(contribs_sorted, dims = 2))
    for i in eachindex(order)
        band!(ax, σE_list, cumulative, cumulative .+ contribs_sorted[:, i],
            color = (colors[i], 0.9))
        cumulative .+= contribs_sorted[:, i]
    end

    overlap_sorted = sv_overlaps[order]
    max_overlap = maximum(overlap_sorted)
    for i in eachindex(order)
        x = sqrt(b) * σ_sorted[i]
        α = 0.15 + 0.85 * overlap_sorted[i] / max_overlap
        lines!(ax, [x, x], [0.0, total_mse_max + 0.02], color = (:black, α),
            linestyle = :dot, linewidth = 2.0)
    end

    lines!(ax, σE_list, cumulative, color = :grey, linewidth = 3, label = "Predicted MSE")

    lines!(ax, σE_list, mse_mat_small[:, idx], color = :black,
        linewidth = 2, label = "Small N MSE")
    lines!(ax, σE_list, mse_mat_large[:, idx], color = :black,
        linestyle = :dash, linewidth = 3, label = "Large N MSE")
    Colorbar(fig[1, 2], limits = (σ_min, σ_max), colormap = cmap,
        label = "Singular value σ_k")
    axislegend(position = :lt)
    display(fig)
end
## ====== Choose system parameters ======
nbr_dots_res = 6
qn_res = 3
sys = tight_binding_system(2, nbr_dots_res, qn_res)
seed = 1234
hams = QDR.matrix_representation_hams(QDR.hamiltonians(sys.grids, seed), sys)
ρ_res = ground_state(hams.res)
t_list = [100]

nbr_states_small = 10^3
nbr_states_large = 10^5
measurements = QDR.charge_measurements(sys)

Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
Ω_small = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main)) for i in 1:nbr_states_small)
Ω_large = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main)) for i in 1:nbr_states_large)

S = scrambling_map(sys, measurements, ρ_res, hams.total, t_list)

## ===== PLOT VARYING NOISE ===========
ps_list = [(:σz, :σz)]
plot_varying_noise(Ω, S, Pm, Pm_dict, ps_list)

## ===== PLOT MODE DECOMPOSITION ===========
S_SVD = svd(S)
b = 0.0147
σE_list = 10 .^ range(-7, 0, length = 200)
mse_mat_small = vcat([get_mse(Ω_small, S, Pm, σE) for σE in σE_list]...)
mse_mat_large = vcat([get_mse(Ω_large, S, Pm, σE) for σE in σE_list]...)
plot_mode_decomposition(mse_mat_small, mse_mat_large, S_SVD, Pm, Pm_dict, (:σy, :σx), b)
