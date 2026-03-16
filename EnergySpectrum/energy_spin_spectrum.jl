using QDReservoir
using LinearAlgebra
import QDReservoir as QDR
using Plots

function get_ham(
        grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_params = QDR.set_dot_params(
        ϵ_func, ϵb_func, u_intra_func, grids.main)
    res_params = QDR.set_dot_params(
        ϵ_func, ϵb_func, u_intra_func, grids.res)
    interaction_params = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(
        grids, main_system_params, res_params, interaction_params)
end

function get_spectrum(ham_tot, sys)
    vals, vecs = eigen(Hermitian(Matrix(ham_tot)))
    s2_op = QDR.total_spin_op(sys.grids.total, sys.H_total)
    s_vals = [QDR.s_from_s2(expectation_value(vecs[:, i], s2_op)) for i in eachindex(vals)]
    return vals, s_vals
end

# ===================== Plot =====================

function plot_energy_spin_spectrum(ham_tot, sys, pl)
    vals, s_vals = get_spectrum(ham_tot, sys)
    plot!(pl, xlim = (0, maximum(s_vals) + 1),
        ylim = (vals[1] - 0.5, vals[end] + 0.5),
        xlabel = "Total Spin",
        ylabel = "Energy",
        title = "$(length(sys.grids.total)) dots and $(QDR.qn_sector(sys.H_total)) electrons",
        legend = false)
    scatter!(pl, s_vals, vals, label = "Spin S")
    hline!(pl, vals, label = "Energy")
    return pl
end

function plot_energy_spin_spectrum_colors(ham_tot, sys, pl)
    vals, s_vals = get_spectrum(ham_tot, sys)
    plot!(pl,
        ylabel = "Energy",
        title = "$(length(sys.grids.total)) dots and $(QDR.qn_sector(sys.H_total)) electrons",
        legend = false)
    xvals = 1:length(vals)
    scatter!(pl,
        xvals,
        vals,
        label = "Spin S",
        marker_z = s_vals,
        color = :viridis,
        marker = :circle,
        markersize = 8)

    return pl
end

# ================ Parameters =====================

ϵ_func() = 0
ϵb_func() = [0, 0, 0]
u_intra_func() = 10
t_func() = 1
t_so_func() = 0
u_inter_func() = 0

nbr_dots_res = 3
qn_res = 3
sys = tight_binding_system(2, nbr_dots_res, qn_res)
hams = QDR.matrix_representation_hams(
    get_ham(sys.grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func),
    sys)

pl = plot()
plot_energy_spin_spectrum_colors(hams.total, sys, pl)
#plot_energy_spin_spectrum(hams.total, sys, pl)
display(pl)