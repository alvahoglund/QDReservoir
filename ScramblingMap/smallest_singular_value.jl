using QDReservoir, LinearAlgebra, CairoMakie, Random
import QDReservoir as QDR

## ===================== Functions ========================
clean_val(y) = map(x -> abs(x) < 1e-10 ? NaN
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
        sv = minimum(svd(S).S)
        sv_sum += sv
        sv_sq_sum += sv^2
    end
    mean = sv_sum / length(randomized_hams)
    std = sqrt.(sv_sq_sum / length(randomized_hams) - mean^2)
    return mean, std
end

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

## ===================== Parameters and system generation ========================
##All parameters
seed = 29084
Random.seed!(seed)

parameters1 = (
    ϵ_func_main = () -> 0.5,
    ϵ_func_res = () -> rand(),
    ϵb_func = () -> [0, 0, 1],
    u_intra_func = () -> 10 + rand(),
    t_func = () -> rand(),
    t_so_func = () -> 0.1 * rand(),
    u_inter_func = () -> rand()
)

nbr_dots_res_list = [2, 3, 4, 5, 6]
t = [100, 200]
nbr_samples = 1

avg_smallest_sv_dict_1 = avg_smallest_sv(nbr_dots_res_list, t, nbr_samples, parameters1)

title = L"H = H_ϵ + H_B + H_{U}^{\text{intra}} +H_{U}^{\text{inter}} + H_t + H_{SO}"
plot_avg_sv_vs_qn(avg_smallest_sv_dict_1, title)
##Removed Uinter
seed = 42879
Random.seed!(seed)

parameters2 = (
    ϵ_func_main = () -> 0.5,
    ϵ_func_res = () -> rand(),
    ϵb_func = () -> [0, 0, 1],
    u_intra_func = () -> 10 + rand(),
    t_func = () -> rand(),
    t_so_func = () -> 0.1 * rand(),
    u_inter_func = () -> 0
)

nbr_dots_res_list = [2, 3, 4, 5, 6]
t = [100, 200]
nbr_samples = 10

avg_smallest_sv_dict_2 = avg_smallest_sv(nbr_dots_res_list, t, nbr_samples, parameters2)
title = L"H = H_ϵ + H_B + H_{U}^{\text{intra}} + H_t + H_{SO}"
plot_avg_sv_vs_qn(avg_smallest_sv_dict_2, title)

##Removed ϵb
seed = 298478
Random.seed!(seed)

parameters3 = (
    ϵ_func_main = () -> 0.5,
    ϵ_func_res = () -> rand(),
    ϵb_func = () -> [0, 0, 0],
    u_intra_func = () -> 10 + rand(),
    t_func = () -> rand(),
    t_so_func = () -> 0.1 * rand(),
    u_inter_func = () -> rand()
)

nbr_dots_res_list = [2, 3, 4, 5, 6]
t = [100, 200]
nbr_samples = 10

avg_smallest_sv_dict_3 = avg_smallest_sv(nbr_dots_res_list, t, nbr_samples, parameters3)
title = L"H = H_ϵ + H_{U}^{\text{intra}} + H_{U}^{\text{inter}} + H_t + H_{SO}"
plot_avg_sv_vs_qn(avg_smallest_sv_dict_3, title)

##Removed SO
seed = 12830
Random.seed!(seed)

parameters4 = (
    ϵ_func_main = () -> 0.5,
    ϵ_func_res = () -> rand(),
    ϵb_func = () -> [0, 0, 1],
    u_intra_func = () -> 10 + rand(),
    t_func = () -> rand(),
    t_so_func = () -> 0,
    u_inter_func = () -> rand()
)

nbr_dots_res_list = [2, 3, 4, 5, 6]
t = [100, 200]
nbr_samples = 10

avg_smallest_sv_dict_4 = avg_smallest_sv(nbr_dots_res_list, t, nbr_samples, parameters4)
title = L"H = H_ϵ + H_B + H_{U}^{\text{intra}} + H_{U}^{\text{inter}} + H_t"
plot_avg_sv_vs_qn(avg_smallest_sv_dict_4, title)
