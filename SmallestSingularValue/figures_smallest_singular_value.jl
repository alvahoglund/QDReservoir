using QDReservoir, CairoMakie, JLD2
import QDReservoir as QDR
includet("plots_smallest_singular_value.jl")

fig_sup = Figure(size = (1200, 800))
## ======== Plot smallest singular value vs_params ========
data_so = load("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_so.jld2")["ssv_dict_so"]
data_ϵb = load("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_eb.jld2")["ssv_dict_eb"]
data_uintra = load("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_uintra.jld2")["ssv_dict_uintra"]
data_uinter = load("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_uinter.jld2")["ssv_dict_uinter"]
data_t = load("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_t.jld2")["ssv_dict_t"]
parameter_range = load("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_uinter.jld2")["u_inter_range"]
datasets = [data_ϵb, data_uintra, data_uinter, data_t, data_so]
labels = [L"\epsilon_b", L"U_\mathrm{intra}", L"U_\mathrm{inter}", L"t", L"t_{SO}"]

function add_param_panel!(gl)
    ax1_param = Axis(gl[1, 1], title = "3 dots, 3 electrons",
        xlabel = "Parameter range", ylabel = "Smallest singular value")
    plot_ssv_param!(ax1_param, (3, 3), datasets, labels, parameter_range)

    ax2_param = Axis(gl[1, 2], title = "6 dots, 3 electrons", xlabel = "Parameter range")
    plot_ssv_param!(ax2_param, (6, 3), datasets, labels, parameter_range)
    axislegend(ax1_param,
        orientation = :horizontal,
        nbanks = 2,
        padding = (4, 4, 4, 4),
        position = :rb,
        colgap = 4,
        rowgap = 2,
        patchlabelgap = 3,
        labelsize = 20
    )
end
fig = Figure(size = (600, 300))
add_param_panel!(fig[1, 1:2])
fig
save("ssv_vs_param.png", fig)

## ======== Plot smallest singular value vs_time ========
data_time = load("SmallestSingularValue/data_vary_time_ssv/avg_sv_dict_time.jld2")["ssv_dict_time"]
data_time_double = load("SmallestSingularValue/data_vary_time_ssv/avg_sv_dict_time_multiplex.jld2")["ssv_dict_time_multiplex"]
time_list = first.(load("SmallestSingularValue/data_vary_time_ssv/avg_sv_dict_time.jld2")["time_list"])

function add_time_panel!(gl)
    ax1_time = Axis(gl[1, 1])
    plot_ssv_time!(
        ax1_time, (6, 3), [data_time, data_time_double], time_list,
        ["Single time, (6, 3)", "Double time, (6, 3)"])

    plot_ssv_time!(ax1_time, (3, 3), [data_time_double], time_list,
        ["Double time, (3, 3)"])
    axislegend(ax1_time, position = :rb, padding = (2, 2, 2, 2))
end
fig = Figure(size = (600, 300))
add_time_panel!(fig[1, 1:2])
fig

## ======= Plot smallest singular value vs qn ========
data_qn = load("SmallestSingularValue/data_vary_qn_ssv/data_vary_qn_ssv_measure_all_dots_100200.jld2")["ssv_dict"]
nbr_dots_res_list = load("SmallestSingularValue/data_vary_qn_ssv/data_vary_qn_ssv_measure_all_dots_100200.jld2")["nbr_dots_res_list"]

function add_qn_panel!(ql)
    ax_qn = Axis(ql, yscale = Makie.Symlog10(1e-5),
        yticks = (
            [0, 1e-6, 1e-4, 10^(-3.5), 1e-3, 10^(-2.5), 1e-2, 1e-1, 1], [
                "0", "10⁻⁶", "10⁻⁴", "10⁻³.⁵", "10⁻³", "10⁻².⁵", "10⁻²", "10⁻¹", "1"]),
        ylabel = "Smallest singular value",
        xlabel = "Electrons relative to half filling")
    plot_ssv_qn!(ax_qn, data_qn[3].median, data_qn[3].std, "3 dots")
    plot_ssv_qn!(ax_qn, data_qn[4].median, data_qn[4].std, "4 dots")
    plot_ssv_qn!(ax_qn, data_qn[5].median, data_qn[5].std, "5 dots")
    plot_ssv_qn!(ax_qn, data_qn[6].median, data_qn[6].std, "6 dots")
    axislegend(ax_qn, position = :rt, orientation = :horizontal,
        tellwidth = false, tellheight = false, nbanks = 2,
        colgap = 4, patchlabelgap = 3, rowgap = 2,
        padding = (4, 4, 4, 4))
end

fig = Figure(size = (600, 300))
add_qn_panel!(fig[1, 1])
fig

## ========= Plot smallest singular value vs multiplexing =========
data_ham_multiplex = load("SmallestSingularValue/data_hamiltonian_multiplex_ssv/data_ham_multiplex_ssv.jld2")["sv_dict_ham_multiplexing"]
data_time_multiplex = load("SmallestSingularValue/data_time_multiplex_ssv/data_time_multiplex_ssv.jld2")["ssv_dict_time_multiplex"]
data_nothing_multiplex = load("SmallestSingularValue/data_hamiltonian_multiplex_ssv/data_nothing_multiplex_ssv.jld2")["sv_dict_nothing_multiplexing"]
data_time_multiplex_late = load("SmallestSingularValue/data_time_multiplex_ssv/data_time_multiplex_ssv_timeframe.jld2")["ssv_dict_time_multiplex"]
times = load("SmallestSingularValue/data_time_multiplex_ssv/data_time_multiplex_ssv.jld2")["times_multiplexing"]

##
fig = Figure(size = (600, 300))
function add_multiplex_panel!(gl)
    ax_multiplex = Axis(gl[1, 1], ylabel = "Smallest singular value",
        xlabel = "M", title = "3 dots, 3 electrons", xscale = log10, yscale = log10)
    ax2_multiplex = Axis(gl[1, 2], xlabel = "M", title = "6 dots, 3 electrons",
        xscale = log10, yscale = log10)

    multiplexing_range = range(1, length(data_ham_multiplex[(3, 3)].median))
    plot_ssv_multiplex!(
        ax_multiplex, multiplexing_range, data_ham_multiplex[(3, 3)], "Hamiltonian", 2)
    plot_ssv_multiplex!(
        ax_multiplex, multiplexing_range, data_time_multiplex[(3, 3)], "Time", 2)

    plot_ssv_multiplex!(ax2_multiplex, multiplexing_range, data_ham_multiplex[(6, 3)], "Hamiltonian")
    plot_ssv_multiplex!(ax2_multiplex, multiplexing_range, data_time_multiplex[(6, 3)], "Time")
    plot_ssv_multiplex!(ax2_multiplex, multiplexing_range, data_nothing_multiplex[(6, 3)], "Identical")

    Legend(gl[1, 1], ax2_multiplex,
        tellwidth = false, tellheight = false,
        halign = :right, valign = :bottom,
        margin = (10, 10, 10, 10))
end
fig = Figure(size = (600, 300))
add_multiplex_panel!(fig[1, 1:2])
fig
save("ssv_multiplexing.png", fig)

##
fig = Figure(size = (600, 800))
add_qn_panel!(fig[1, 1])
add_param_panel!(fig[2, 1:2])
add_time_panel!(fig[1, 2])
add_multiplex_panel!(fig[3, 1:2])
fig
save("ssv_all.png", fig)