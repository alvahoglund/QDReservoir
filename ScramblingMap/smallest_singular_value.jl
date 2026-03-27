using QDReservoir, LinearAlgebra, CairoMakie, Random
import QDReservoir as QDR

## ===================== Functions ========================
clean_val(y) = map(x -> abs(x) < 1e-14 ? NaN
                        : x, y)

function get_ham(grids, ϵ_func_main, ϵ_func_res, ϵb_func,
        u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_parameters = QDR.set_dot_params(
        ϵ_func_main, ϵb_func, u_intra_func, grids.main)
    reservoir_parameters = QDR.set_dot_params(ϵ_func_res, ϵb_func, u_intra_func, grids.res)
    interaction_parameters = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(grids, main_system_parameters,
        reservoir_parameters, interaction_parameters)
end

function get_ham(grids, parameters)
    get_ham(grids, parameters.ϵ_func_main, parameters.ϵ_func_res, parameters.ϵb_func,
        parameters.u_intra_func, parameters.t_func, parameters.t_so_func, parameters.u_inter_func)
end

function avg_smallest_sv(grid, qn_res, randomized_hams, t, measurements)
    sys = tight_binding_system(grid, qn_res)
    sv_sum = 0.0
    sv_sq_sum = 0.0
    for i in eachindex(randomized_hams)
        println("Calculating for reservoir electrons: $(qn_res), sample: $(i)")
        hams = QDR.matrix_representation_hams(randomized_hams[i], sys)
        S = scrambling_map(sys, QDR.matrix_representation_ops(measurements, sys.H_total),
            ground_state(hams.res), hams.total, t)
        sv = minimum(svdvals(S))
        sv_sum += sv
        sv_sq_sum += sv^2
    end
    mean = sv_sum / length(randomized_hams)
    std = sqrt.(sv_sq_sum / length(randomized_hams) - mean^2)
    return mean, std
end

## ===================== Functions for Singular values vs. reservoir electrons ========================
function avg_smallest_sv(nbr_dots_res_list, t, nbr_samples, parameters)
    nbr_dots_main = 2
    avg_smallest_sv_dict = Dict{Tuple{Int, Int}, Tuple{Float64, Float64}}()
    for nbr_dots_res in nbr_dots_res_list
        println("Calculating for reservoir dots: $(nbr_dots_res)")
        grid = QDR.generate_grid(nbr_dots_main, nbr_dots_res)
        measurements = QDR.charge_probabilities(grid.total)
        randomized_hams = [get_ham(grid, parameters) for _ in 1:nbr_samples]
        for qn_res in 0:(2 * nbr_dots_res)
            mean, std = avg_smallest_sv(grid, qn_res, randomized_hams, t, measurements)
            avg_smallest_sv_dict[(nbr_dots_res, qn_res)] = (mean, std)
        end
    end
    return avg_smallest_sv_dict
end

function plot_avg_sv_vs_qn(nbr_dots_res, sv_qn, ax)
    means = clean_val.(getindex.(sv_qn, 1))

    stds = clean_val.(getindex.(sv_qn, 2))
    x = 0:(length(means) - 1)

    lower = means
    upper = means .+ stds
    CairoMakie.band!(ax, x, lower, upper, alpha = 0.3)
    CairoMakie.lines!(ax, x, means)
    CairoMakie.scatter!(ax, x, means, label = "res dots: $(nbr_dots_res)")
end

function plot_avg_sv_vs_qn(avg_smallest_sv_dict, title)
    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1],
        ylabel = "Average smallest singular value",
        xlabel = "Electrons in reservoir",
        title = title,
        titlesize = 24,
        yscale = log10
    )

    for nbr_dots_res in nbr_dots_res_list
        sv_qn = [avg_smallest_sv_dict[(nbr_dots_res, qn)] for qn in 0:(2 * nbr_dots_res)]
        plot_avg_sv_vs_qn(nbr_dots_res, sv_qn, ax)
    end
    CairoMakie.axislegend(ax, position = :rb)
    CairoMakie.display(fig)
end

## ===================== Functions for Singular values vs parameter size ========================
function avg_sv_vs_param(nbr_dots_res, qn_res, parameter_list, nbr_samples, t)
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = QDR.charge_probabilities(grid.total)

    randomized_hams_list = [[get_ham(grid, parameters) for _ in 1:nbr_samples]
                            for parameters in parameter_list]
    avg_sv_list = [avg_smallest_sv(grid, qn_res, randomized_hams, t, measurements)
                   for randomized_hams in randomized_hams_list]
    return avg_sv_list
end

function avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)
    avg_sv_dict = Dict{Tuple{Int, Int}, Vector{Tuple{Float64, Float64}}}()
    for (nbr_dots_res, qn_res) in reservoir_settings
        println("Calculating for reservoir dots: $(nbr_dots_res), reservoir electrons: $(qn_res)")
        avg_sv_list = avg_sv_vs_param(nbr_dots_res, qn_res, parameters_list, nbr_samples, t)
        avg_sv_dict[(nbr_dots_res, qn_res)] = avg_sv_list
    end
    return avg_sv_dict
end

function plot_avg_sv_vs_param(avg_sv_dict, x_range, title, xlabel)
    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1],
        ylabel = "Average smallest singular value",
        xlabel = xlabel,
        title = title,
        titlesize = 18,
        yscale = log10,
        xscale = log10
    )
    for (nbr_dots_res, qn_res) in keys(avg_sv_dict)
        avg_sv_list = avg_sv_dict[(nbr_dots_res, qn_res)]

        means = clean_val.(getindex.(avg_sv_list, 1))
        stds = clean_val.(getindex.(avg_sv_list, 2))

        lower = means
        upper = means .+ stds
        CairoMakie.band!(ax, x_range, lower, upper, alpha = 0.3)
        CairoMakie.lines!(ax, x_range, means)
        CairoMakie.scatter!(ax, x_range, means,
            label = "res dots: $(nbr_dots_res), res electrons: $(qn_res)")
    end
    CairoMakie.axislegend(ax, position = :rb)
    CairoMakie.display(fig)
end
