using QDReservoir
using LinearAlgebra
import QDReservoir as QDR
using Plots

function get_ham(
        nbr_dots_res, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    sys_temp = tight_binding_system(2, nbr_dots_res, 0)
    main_system_parameters = QDR.set_dot_params(
        ϵ_func, ϵb_func, u_intra_func, sys_temp.grid.main)
    reservoir_parameters = QDR.set_dot_params(
        ϵ_func, ϵb_func, u_intra_func, sys_temp.grid.res)
    interaction_parameters = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, sys_temp.grid.total)
    hamiltonians(
        sys_temp, main_system_parameters, reservoir_parameters, interaction_parameters)
end

function get_ground_state(hams, sys)
    ψmain = def_state(singlet, sys.H_main, sys.f)
    ψres = ground_state(hams.hamiltonian_reservoir)
    ψtot = generalized_kron((ψmain, ψres), (sys.H_main, sys.H_reservoir) => sys.H_total)
    E_exp = QDR.expectation_value(ψtot, hams.hamiltonian_total)
    return ψtot, E_exp
end

energy_amplitudes(ψtot, vecs) = [abs2(dot(vecs[:, i], ψtot)) for i in 1:size(vecs)[1]]

# ================== Plot ===========================

function get_plot_index(nbr_dots_res, qn_res)
    if qn_res ≤ nbr_dots_res
        col = 1
        row = qn_res + 1
    else
        col = 2
        row = 2 * (nbr_dots_res + 1) - (qn_res + 1)
    end
    return row, col
end

function plot_data!(p_es, subplot_idx, vals, E_exp, c_i2, qn_res)
    xvals = 1:length(vals)
    ymin, ymax = minimum(vals) - 0.1, maximum(vals) + 0.1
    background = repeat(reshape(c_i2[xvals], 1, :), 100, 1)

    # Background heatmap in the subplot
    heatmap!(p_es, xvals, LinRange(ymin, ymax, 100), background,
        color = cgrad([:white, :purple]),
        legend = false,
        ylims = (ymin, ymax),
        subplot = subplot_idx)

    scatter!(p_es, xvals, vals[xvals], color = :white, ms = 5,
        title = "Electrons in reservoir: $(qn_res)",
        subplot = subplot_idx)

    hline!(p_es, [E_exp],
        subplot = subplot_idx,
        color = :grey,
        linestyle = :dash,
        linewidth = 5)
end

function plot_spectrum(hams_s, nbr_dots_res)
    p_es = plot(layout = (nbr_dots_res + 1, 2), size = (1100, 1100))

    for qn_res in 0:(2 * nbr_dots_res)
        sys = tight_binding_system(2, nbr_dots_res, qn_res)
        hams_m = QDR.matrix_representation_hams(hams_s, sys)

        row, col = get_plot_index(nbr_dots_res, qn_res)
        subplot_idx = (row - 1) * 2 + col

        vals, vecs = eigen(Matrix(hams_m.hamiltonian_total))
        ψtot, E_exp = get_ground_state(hams_m, sys)
        c_i2 = energy_amplitudes(ψtot, vecs)

        plot_data!(p_es, subplot_idx, vals, E_exp, c_i2, qn_res)
    end
    plot!(p_es,
        suptitle = " Energy spectrum of Hamiltonian block \n Dots in reservoir: $(nbr_dots_res)",
        ylabel = "Eigenenergy")
    display(p_es)
end

# ============== PARAMETERS =====================

nbr_dots_res = 1
ϵ_func() = 0
ϵb_func() = [0, 0, 0]
u_intra_func() = 10

t_func() = 1
t_so_func() = 0
u_inter_func() = 0
ham_tot = get_ham(
    nbr_dots_res, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)

plot_spectrum(ham_tot, nbr_dots_res)
