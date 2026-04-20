includet("smallest_singular_value.jl")
using CairoMakie, Random
using CairoMakie, Random

function plot_avg_sv_vs_param(avg_sv_dict, x_range, title, xlabel)
    nbr_dots_res_list = sort(unique(getindex.(keys(avg_sv_dict), 1)))

    #fig = Figure(size = (700, 1000))
    fig = Figure()
    axs = [Axis(fig[i + 1, 1],
               ylabel = "Average smallest singular value",
               xlabel = xlabel,
               title = "Nbr dots in reservoir: $(nbr_dots_res_list[i]),  $title ",
               titlesize = 18,
               yscale = log10,
               xscale = log10,
               xticks = LogTicks(WilkinsonTicks(6, k_min = 5)),
               yticks = LogTicks(WilkinsonTicks(6, k_min = 5)))
           for i in eachindex(nbr_dots_res_list)]
    sorted_keys = sort(collect(keys(avg_sv_dict)), by = x -> x[2])

    for (nbr_dots_res, qn_res) in sorted_keys
        avg_sv_list = avg_sv_dict[(nbr_dots_res, qn_res)]

        means = clean_val.(avg_sv_list[1])
        stds = clean_val.(avg_sv_list[2])

        lower = means
        upper = means .+ stds
        ax = axs[findfirst(==(nbr_dots_res), nbr_dots_res_list)]
        ax = axs[findfirst(==(nbr_dots_res), nbr_dots_res_list)]
        CairoMakie.band!(ax, x_range, lower, upper, alpha = 0.3)
        CairoMakie.lines!(ax, x_range, means)
        CairoMakie.scatter!(ax, x_range, means,
            label = "$(qn_res)")
    end
    fig[1, 1] = Legend(fig, axs[end], "Electrons in reservoir",
        orientation = :horizontal, framevisible = false)

    CairoMakie.display(fig)
    return fig
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

tso_range = 10 .^ range(-3, 1, length = 10)
tso_range = 10 .^ range(-3, 1, length = 10)
parameters_list = [parameters_vary_so(tso) for tso in tso_range]
reservoir_settings = [
    (3, 0), (3, 1), (3, 2), (3, 3), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1),
    (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
nbr_samples = 20
t = [100, 200]

@time avg_sv_dict = avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)

title = "Vary Spin Orbit Coupling"
title = "Vary Spin Orbit Coupling"
xlabel = "SO coupling strength (relative)"
plot_avg_sv_vs_param(avg_sv_dict, tso_range, title, xlabel)

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

t_range = 10 .^ range(-3, 2, length = 20)
parameters_list = [parameters_vary_so(t) for t in t_range]
#reservoir_settings = [(3,0), (3,1), (3,2), (3,3), (2,0), (2,1), (2,2), (4, 0), (4,1), (4,2), (4,3), (4,4), (5,0), (5,1), (5,2), (5,3), (5, 4), (5,5)]
reservoir_settings = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
nbr_samples = 20
t = [100, 200]

@time avg_sv_dict = avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)

title = "Time: $(t), tso: 1*rand()"
xlabel = "Tunneling coupling strength"
plot_avg_sv_vs_param(avg_sv_dict, t_range, title, xlabel)

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

ϵb_range = 10 .^ range(-10, 1, length = 20)
ϵb_range = 10 .^ range(-10, 1, length = 20)
parameters_list = [parameters_vary_ϵb(ϵb) for ϵb in ϵb_range]
reservoir_settings = [
    (3, 0), (3, 1), (3, 2), (3, 3), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1),
    (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
nbr_samples = 20
t = [100, 200]
avg_sv_dict = avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)
title = "Vary applied magnetic field"
xlabel = "Zeeman splitting of all dots"
title = "Vary applied magnetic field"
xlabel = "Zeeman splitting of all dots"
plot_avg_sv_vs_param(avg_sv_dict, ϵb_range, title, xlabel)

## ================== Vary u_intra ==================
seed = 9823
Random.seed!(seed)

function parameters_vary_u_intra(u_intra)
    (
        ϵ_func_main = () -> 0.5,
        ϵ_func_res = () -> rand(),
        ϵb_func = () -> [0, 0, 1],
        u_intra_func = () -> u_intra * (10 + rand()),
        u_intra_func = () -> u_intra * (10 + rand()),
        t_func = () -> rand(),
        t_so_func = () -> 0.1 * rand(),
        u_inter_func = () -> rand()
    )
end

u_intra_range = 10 .^ range(-2, 1, length = 20)
u_intra_range = 10 .^ range(-2, 1, length = 20)
parameters_list = [parameters_vary_u_intra(u_intra) for u_intra in u_intra_range]
reservoir_settings = [
    (3, 0), (3, 1), (3, 2), (3, 3), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1),
    (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
nbr_samples = 20
t = [100, 200]

@time avg_sv_dict = avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)

title = "U_intra = k * (10 + rand())"
title = "U_intra = k * (10 + rand())"
xlabel = "Intra-dot interaction strength (k)"
fig = plot_avg_sv_vs_param(avg_sv_dict, u_intra_range, title, xlabel)

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

u_inter_range = 10 .^ range(-3, 1, length = 20)
parameters_list = [parameters_vary_u_inter(u_inter) for u_inter in u_inter_range]
reservoir_settings = [
    (3, 0), (3, 1), (3, 2), (3, 3), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1),
    (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
nbr_samples = 20
t = [100, 200]

@time avg_sv_dict = avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)

title = "U_inter = k * rand())"
xlabel = "Inter-dot interaction strength (k)"
fig = plot_avg_sv_vs_param(avg_sv_dict, u_inter_range, title, xlabel)
