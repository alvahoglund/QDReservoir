using QDReservoir, LinearAlgebra, CairoMakie
import QDReservoir as QDR

## ===================== Functions ========================
clean_val(y) = map(x -> abs(x) < 1e-10 ? 0.0
                        : x, y)

function get_ham(grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.main)
    reservoir_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.res)
    interaction_parameters = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(grids, main_system_parameters,
        reservoir_parameters, interaction_parameters)
end

function avg_smallest_sv(grid, qn_res, nbr_samples)
    sys = tight_binding_system(grid, qn_res)
    sv_sum = 0.0
    sv_sq_sum = 0.0
    for i in 1:nbr_samples
        hams = QDR.matrix_representation_hams(QDR.hamiltonians(grid), sys)
        S = scrambling_map(sys, QDR.matrix_representation_ops(measurements, sys.H_total),
            ground_state(hams.res), hams.total, t)
        sv = minimum(svd(S).S)
        sv_sum += sv
        sv_sq_sum += sv^2
    end
    mean = sv_sum / nbr_samples
    stds = sqrt.(sv_sq_sum / nbr_samples - mean^2)
    return mean, stds
end

function plot_avg_sv_vs_qn(sv_qn)
    fig = Figure(title = "Average smallest singular value vs electrons in reservoir",
        size = (700, 500))
    ax = Axis(fig[1, 1], yscale = log10, ylabel = "Average smallest singular value",
        xlabel = "Electrons in reservoir")
    means = getindex.(sv_qn, 1)
    stds = getindex.(sv_qn, 2)
    x = 0:(length(means) - 1)

    lower = max.(means .- stds, eps())
    upper = means .+ stds

    band!(ax, x, lower, upper, color = (:blue, 0.2), label = "±1 std dev")
    lines!(ax, x, means, color = :blue)
    scatter!(ax, x, means, color = :blue, label = "Mean smallest singular value")
    axislegend(ax, position = :rb)
    display(fig)
end

## ===================== Parameters and system generation ========================

nbr_dots_main = 2
nbr_dots_res = 3
grid = QDR.generate_grid(nbr_dots_main, nbr_dots_res)
t = [100, 200, 300]
measurements = QDR.charge_probabilities(grid.total)

qn_res = 4

nbr_samples = 5
avg_sv = avg_smallest_sv(grid, qn_res, nbr_samples)
sv_qn = [avg_smallest_sv(grid, qn, nbr_samples) for qn in 0:(2 * nbr_dots_res)]

##
plot_avg_sv_vs_qn(sv_qn)
