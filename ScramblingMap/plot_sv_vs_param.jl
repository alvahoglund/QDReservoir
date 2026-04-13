includet("smallest_singular_value.jl")
using CairoMakie

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

        means = clean_val.(avg_sv_list[1])
        stds = clean_val.(avg_sv_list[2])

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

tso_range = 10 .^ range(-3, 0.1, length = 10)
parameters_list = [parameters_vary_so(tso) for tso in tso_range]
reservoir_settings = [(2, 1), (4, 3)]
nbr_samples = 10
t = [100, 200]

@time avg_sv_dict = avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)

title = "Average smallest singular value vs. SO coupling strength"
xlabel = "SO coupling strength (relative)"
plot_avg_sv_vs_param(avg_sv_dict, tso_range, title, xlabel)

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

ϵb_range = 10 .^ range(-10, -1, length = 20)
parameters_list = [parameters_vary_ϵb(ϵb) for ϵb in ϵb_range]
reservoir_settings = [(2, 1), (3, 3), (4, 3)]
nbr_samples = 10
t = [100, 200]
avg_sv_dict = avg_sv_vs_param(reservoir_settings, parameters_list, nbr_samples, t)
title = "Average smallest singular value vs. zeeman splitting of reservoir dots"
xlabel = "Maximum zeeman splitting of reservoir dots"
plot_avg_sv_vs_param(avg_sv_dict, ϵb_range, title, xlabel)