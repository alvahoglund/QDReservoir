using QDReservoir, LinearAlgebra, CairoMakie
import QDReservoir as QDR

## ===================== Functions ======================== 
function plot_sv_against_time(S_list, t_list)
    S_svd_list = map(svd, S_list)
    S_sv_list = stack(map(S_SVD -> S_SVD.S, S_svd_list))

    fig = Figure(title = "Singular values of scrambling map over time")
    ax = Axis(fig[1, 1], xlabel = "Time", ylabel = "Singular value",
        title = "Singular values over time", xscale = log10, yscale = log10)

    for i in axes(S_sv_list, 1)[2:end]
        lines!(ax, t_list, S_sv_list[i, :], label = "σ$(i)")
    end

    display(fig)
end

function plot_sv_against_time_multiplexing(S_list, t_list)
    tm = range(2, length(t_list))
    S_list_tm = [reduce(vcat, S_list[1:i]) for i in tm]
    S_svd_list_tm = hcat([svd(Smap).S for Smap in S_list_tm]...)

    fig = Figure()
    ax = Axis(fig[1, 1],
        ylabel = "Singular values",
        xlabel = "Time multiplexing",
        title = "Random hamiltonian without SO, res dots:$(nbr_dots_res), res electrons: $(qn_res), \n linear time multiplexing ($(t_list[1]) -$(t_list[end]))")
    for i in 3:15
        lines!(ax, tm, S_svd_list_tm[i, :], label = "σ$(i)")
    end

    display(fig)
end

clean_val(y) = map(x -> abs(x) < 1e-10 ? 0.0 : x, y)

function get_ham(grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.main)
    reservoir_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.res)
    interaction_parameters = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(grids, main_system_parameters,
        reservoir_parameters, interaction_parameters)
end

## ===================== Parameters and system generation ========================
nbr_dots_main = 2
nbr_dots_res = 6
qn_res = 2
sys = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_res)

ϵ_func() = rand()
ϵb_func() = [0, 0, 3 * rand()]
u_intra_func() = 10 * rand()
t_func() = 1 * rand()
t_so_func() = 0
u_inter_func() = 1 * rand()

hams = QDR.matrix_representation_hams(
    get_ham(sys.grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func), sys)
#hams = QDR.matrix_representation_hams(QDR.hamiltonians_equal_param(sys.grids), sys)
measurements = QDR.charge_probabilities(sys)

## Scrambling maps over time

t_list = 10 .^ range(-1, 2, length = 50)
S_list = map(
    t -> scrambling_map(sys, measurements, ground_state(hams.res), hams.total, t), t_list)

## Plot over time
plot_sv_against_time(S_list, t_list)
plot_sv_against_time_multiplexing(S_list, t_list)