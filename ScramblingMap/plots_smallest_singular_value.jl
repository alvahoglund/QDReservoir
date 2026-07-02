function plot_ssv_param_line!(ax, parameter_range, data_median, data_std, label)
    CairoMakie.band!(
        ax, parameter_range, data_median, data_median .+ data_std, alpha = 0.3)
    CairoMakie.lines!(ax, parameter_range, data_median, label = label)
    CairoMakie.scatter!(ax, parameter_range, data_median, markersize = 6)
end

function plot_ssv_param!(ax, setting, datasets, labels, parameter_range)
    ax.yscale = Makie.Symlog10(1e-8)
    ax.yticks = ([0, 1e-6, 1e-4, 1e-2, 1e-1, 1],
        ["0", "10⁻⁶", "10⁻⁴", "10⁻²", "10⁻¹", "1"])
    for (data, label) in zip(datasets, labels)
        med = data[setting].median
        std = data[setting].std
        plot_ssv_param_line!(ax, parameter_range, med, std, label)
    end
end

function plot_ssv_time_line!(ax, time_list, data_median, data_std, label)
    CairoMakie.band!(
        ax, time_list, data_median, data_median .+ data_std, alpha = 0.3)
    CairoMakie.lines!(ax, time_list, data_median, label = label)
    CairoMakie.scatter!(ax, time_list, data_median, markersize = 6)
end

function plot_ssv_time!(ax, setting, datasets, time_list, labels)
    ax.xlabel = "Time (s)"
    ax.ylabel = "Smallest singular value"
    ax.yscale = Makie.Symlog10(1e-8)
    ax.yticks = ([0, 1e-6, 1e-4, 1e-2, 1e-1, 1],
        ["0", "10⁻⁶", "10⁻⁴", "10⁻²", "10⁻¹", "1"])
    for (data, label) in zip(datasets, labels)
        med = data[setting].median
        std = data[setting].std
        plot_ssv_time_line!(ax, time_list, med, std, label)
    end
end

function plot_ssv_qn!(ax, median_sv, std_sv, label)
    # x runs over 0:2*nbr_dots electrons; center on half filling (the midpoint)
    half_filling = (length(median_sv) - 1) / 2
    x = (0:(length(median_sv) - 1)) .- half_filling
    CairoMakie.band!(ax, x, median_sv, median_sv .+ std_sv, alpha = 0.3)
    CairoMakie.lines!(ax, x, median_sv, label = label)
    CairoMakie.scatter!(ax, x, median_sv, markersize = 6)
end

function fit_sqrt(x, y; fit_from = 1)
    10.0^(sum(log10.(y[fit_from:end]) .- 0.5 .* log10.(x[fit_from:end])) /
          (length(x) - fit_from + 1))
end

function plot_ssv_multiplex!(ax, multiplexing_range, data_dict, label, plot_from = 1)
    CairoMakie.lines!(ax, multiplexing_range[plot_from:end], data_dict.median[plot_from:end], label = label)
    CairoMakie.scatter!(ax, multiplexing_range[plot_from:end], data_dict.median[plot_from:end])
    CairoMakie.band!(
        ax, multiplexing_range[plot_from:end], data_dict.median[plot_from:end],
        data_dict.median[plot_from:end] .+ data_dict.std[plot_from:end], alpha = 0.3)
    y = sqrt.(multiplexing_range[plot_from:end])
    m = fit_sqrt(multiplexing_range, data_dict.median; fit_from = 70)
    CairoMakie.lines!(
        ax, multiplexing_range[plot_from:end], m .* y, color = :black, linestyle = :dash)
end
