function plot_varying_noise!(
        gl, σE_list, mse_list, mse_pred_list, S_SVD, sv_overlaps, Pm_dict, ps_list, b)
    first_ax = nothing
    for (i, ps) in enumerate(ps_list)
        idx = Pm_dict[ps...]
        ax = Axis(gl[1, i], xlabel = "Noise level (σE)", ylabel = "MSE",
            title = "Estimating $(ps[1]) ⊗ $(ps[2])", xscale = log10,
            xgridvisible = false, ygridvisible = false)
        lines!(ax, σE_list, mse_list[:, idx], label = "MSE")
        lines!(ax, σE_list, mse_pred_list[:, idx], label = "Predicted \nMSE")
        vlines!(ax, sqrt(b) .* S_SVD.S, color = sv_overlaps[:, idx],
            colormap = :Blues, colorrange = (-0.5, maximum(sv_overlaps[:, idx])),
            linestyle = :dash, label = "√b·σS")
        i == 1 && (first_ax = ax)
    end
    axislegend(first_ax, position = :lt, framevisible = false)
end

function plot_varying_noise(
        σE_list, mse_list, mse_pred_list, S_SVD, sv_overlaps, Pm_dict, ps_list, b)
    fig = Figure(size = (600, 250))
    plot_varying_noise!(fig.layout, σE_list, mse_list, mse_pred_list,
        S_SVD, sv_overlaps, Pm_dict, ps_list, b)
    return fig
end

function plot_mode_decomposition!(
        gl, σE_list, mse_mat_small, mse_mat_large, S_SVD, Pm, Pm_dict, ps, b)
    idx = Pm_dict[ps...]
    sv_overlaps = SV_overlap(S_SVD, Pm)[:, idx]

    contributions = hcat([(b * σE^2) ./ (b .* S_SVD.S .^ 2 .+ σE^2) .* sv_overlaps
                          for σE in σE_list]...)'

    order = sortperm(S_SVD.S)
    contribs_sorted = contributions[:, order]
    σ_sorted = S_SVD.S[order]

    ax = Axis(gl[1, 1], xlabel = "Noise level (σE)", ylabel = "MSE contribution",
        title = "MSE of predicting $(ps[1]) ⊗ $(ps[2])", xscale = log10,
        xgridvisible = false, ygridvisible = false)

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
    Colorbar(
        gl[1, 2], limits = (σ_min, σ_max), colormap = cmap, label = "Singular value σ_k")
    axislegend(ax, position = :lt)
end

function plot_mode_decomposition(
        σE_list, mse_mat_small, mse_mat_large, S_SVD, Pm, Pm_dict, ps, b)
    fig = Figure(size = (700, 300))
    plot_mode_decomposition!(
        fig.layout, σE_list, mse_mat_small, mse_mat_large, S_SVD, Pm, Pm_dict, ps, b)
    return fig
end
