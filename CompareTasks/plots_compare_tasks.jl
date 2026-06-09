
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
