includet("smallest_singular_value.jl")
using CairoMakie, Random, JLD2

##

function plot_avg_sv_vs_param(avg_sv_dict, x_range, title, xlabel)
    nbr_dots_res_list = sort(unique(getindex.(keys(avg_sv_dict), 1)))

    fig = Figure(size = (700, 900))
    axs = [Axis(fig[i + 1, 1],
               ylabel = "Median smallest SV",
               xlabel = xlabel,
               title = "Nbr dots in reservoir: $(nbr_dots_res_list[i]),  $title ",
               titlesize = 18,
               yscale = Makie.Symlog10(1e-10),
               yticks = ([0, 1e-6, 1e-4, 1e-2, 1e-1, 1],
                   ["0", "10⁻⁶", "10⁻⁴", "10⁻²", "10⁻¹", "1"]))
           for i in eachindex(nbr_dots_res_list)]
    sorted_keys = sort(collect(keys(avg_sv_dict)), by = x -> x[2])

    for (nbr_dots_res, qn_res) in sorted_keys
        avg_sv_list = avg_sv_dict[(nbr_dots_res, qn_res)]

        means = avg_sv_list[1]
        stds = avg_sv_list[2]
        medians = avg_sv_list[3]

        lower = medians
        upper = medians .+ stds
        ax = axs[findfirst(==(nbr_dots_res), nbr_dots_res_list)]
        CairoMakie.band!(ax, x_range, lower, upper, alpha = 0.3)
        CairoMakie.lines!(ax, x_range, medians)
        CairoMakie.scatter!(ax, x_range, medians,
            label = "$(qn_res)")
    end
    fig[1, 1] = Legend(fig, axs[end], "Electrons in reservoir",
        orientation = :horizontal, framevisible = false)

    CairoMakie.display(fig)
    return fig
end

## ================== Vary SO ==================
seed = 42899
Random.seed!(seed)

function parameters_vary_so(tso)
    (
        ϵ_func_main = () -> 0.5,
        ϵ_func_res = () -> rand(),
        ϵb_func = () -> [0, 0, 1],
        u_intra_func = () -> 10 + rand(),
        t_func = () -> rand(),
        t_so_func = () -> tso * rand(),
        u_inter_func = () -> rand()
    )
end

tso_range = range(0, 1, length = 20)
parameters_list = [parameters_vary_so(tso) for tso in tso_range]
reservoir_settings = [
    (3, 0), (3, 1), (3, 2), (3, 3), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1),
    (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]

nbr_samples = 20
t = [100, 200]

@time avg_sv_dict_so = avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)
#jldsave("ScramblingMap/sv_vs_param_data/avg_sv_dict_so_temp.jld2";
#    avg_sv_dict_so, tso_range, nbr_samples, t)

##
data = load("ScramblingMap/sv_vs_param_data/avg_sv_dict_so.jld2")
title = "Vary Spin Orbit Coupling"
xlabel = "SO coupling strength (relative)"
plot_avg_sv_vs_param(data["avg_sv_dict_so"], data["tso_range"], title, xlabel)

## ================== Vary t ===================
seed = 23498
Random.seed!(seed)

function parameters_vary_t(t)
    (
        ϵ_func_main = () -> 0.5,
        ϵ_func_res = () -> rand(),
        ϵb_func = () -> [0, 0, 1],
        u_intra_func = () -> 10 + rand(),
        t_func = () -> t * rand(),
        t_so_func = () -> 0.1 * rand(),
        u_inter_func = () -> rand()
    )
end

t_range = range(0, 10, length = 20)
parameters_list = [parameters_vary_so(t) for t in t_range]
reservoir_settings = [
    (3, 0), (3, 1), (3, 2), (3, 3), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1),
    (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
nbr_samples = 20
t = [100, 200]

@time avg_sv_dict_t = avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)
#jldsave("ScramblingMap/sv_vs_param_data/avg_sv_dict_t.jld2";
#    avg_sv_dict_t, t_range, nbr_samples, t)

##
data = load("ScramblingMap/sv_vs_param_data/avg_sv_dict_t.jld2")

title = "Vary tunneling"
xlabel = "Tunneling coupling strength"
plot_avg_sv_vs_param(data["avg_sv_dict_t"], data["t_range"], title, xlabel)

## ================== Vary ϵb ==================
seed = 30298
Random.seed!(seed)

function parameters_vary_ϵb(ϵb)
    (
        ϵ_func_main = () -> 0.5,
        ϵ_func_res = () -> rand(),
        ϵb_func = () -> [0, 0, ϵb],
        u_intra_func = () -> 10 + rand(),
        t_func = () -> rand(),
        t_so_func = () -> 0.1 * rand(),
        u_inter_func = () -> rand()
    )
end

ϵb_range = range(0, 5, length = 20)
parameters_list = [parameters_vary_ϵb(ϵb) for ϵb in ϵb_range]
reservoir_settings = [
    (3, 0), (3, 1), (3, 2), (3, 3), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1),
    (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
nbr_samples = 20
t = [100, 200]
@time avg_sv_dict_eb = avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)
#jldsave("ScramblingMap/sv_vs_param_data/avg_sv_dict_eb.jld2";
#    avg_sv_dict_eb, ϵb_range, nbr_samples, t)

##
data = load("ScramblingMap/sv_vs_param_data/avg_sv_dict_eb.jld2")

title = "Vary applied magnetic field"
xlabel = "Zeeman splitting of all dots"
plot_avg_sv_vs_param(data["avg_sv_dict_eb"], data["ϵb_range"], title, xlabel)

## ================== Vary u_intra ==================
seed = 9823
Random.seed!(seed)

function parameters_vary_u_intra(u_intra)
    (
        ϵ_func_main = () -> 0.5,
        ϵ_func_res = () -> rand(),
        ϵb_func = () -> [0, 0, 1],
        u_intra_func = () -> u_intra * (10 + rand()),
        t_func = () -> rand(),
        t_so_func = () -> 0.1 * rand(),
        u_inter_func = () -> rand()
    )
end

u_intra_range = range(0, 10, length = 30)
parameters_list = [parameters_vary_u_intra(u_intra) for u_intra in u_intra_range]
reservoir_settings = [
    (3, 0), (3, 1), (3, 2), (3, 3), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1),
    (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
nbr_samples = 20
t = [100, 200]

@time avg_sv_dict_uintra = avg_sv_vs_param(
    reservoir_settings, parameters_list, nbr_samples, t)
#jldsave("ScramblingMap/sv_vs_param_data/avg_sv_dict_uintra.jld2";
#    avg_sv_dict_uintra, u_intra_range, nbr_samples, t)

##
data = load("ScramblingMap/sv_vs_param_data/avg_sv_dict_uintra.jld2")

title = "U_intra = k * (10 + rand())"
xlabel = "Intra-dot interaction strength (k)"
fig = plot_avg_sv_vs_param(data["avg_sv_dict_uintra"], data["u_intra_range"], title, xlabel)

## ================== Vary u_inter ==================
seed = 9823
Random.seed!(seed)

function parameters_vary_u_inter(u_inter)
    (
        ϵ_func_main = () -> 0.5,
        ϵ_func_res = () -> rand(),
        ϵb_func = () -> [0, 0, 1],
        u_intra_func = () -> (10 + rand()),
        t_func = () -> rand(),
        t_so_func = () -> 0.1 * rand(),
        u_inter_func = () -> u_inter * rand()
    )
end

u_inter_range = range(0, 10, length = 30)
parameters_list = [parameters_vary_u_inter(u_inter) for u_inter in u_inter_range]
reservoir_settings = [
    (3, 0), (3, 1), (3, 2), (3, 3), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1),
    (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
nbr_samples = 20
t = [100, 200]

@time avg_sv_dict_uinter = avg_sv_vs_param(
    reservoir_settings, parameters_list, nbr_samples, t)
#jldsave("ScramblingMap/sv_vs_param_data/avg_sv_dict_uinter.jld2";
#    avg_sv_dict_uinter, u_inter_range, nbr_samples, t)

##
data = load("ScramblingMap/sv_vs_param_data/avg_sv_dict_uinter.jld2")

title = "U_inter = k * rand())"
xlabel = "Inter-dot interaction strength (k)"
fig = plot_avg_sv_vs_param(data["avg_sv_dict_uinter"], data["u_inter_range"], title, xlabel)
