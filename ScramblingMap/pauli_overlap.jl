using QDReservoir, LinearAlgebra, CairoMakie
import QDReservoir as QDR

##  ==================== Functions ========================

function plot_Pauli_overlap_over_time(t_list, S_pauli_overlaps, Pm_dict)
    fig = Figure(title = "Pauli string overlap over time", size = (700, 700))
    for ps in keys(Pm_dict)
        idx = Pm_dict[ps...]
        ax = Axis(fig[(idx - 1) ÷ 4 + 1, (idx - 1) % 4 + 1], title = "$(ps[1]) ⊗ $(ps[2])")
        lines!(ax, t_list, clean_val.(S_pauli_overlaps[:, idx]))
    end
    display(fig)
end

ps_labels = [("$(a) ⊗ $(b)")
             for a in ["σ0", "σx", "σy", "σz"], b in ["σ0", "σx", "σy", "σz"]]

function test_contains(S, ps)
    rank(Matrix(vcat(S, ps')), rtol = 1e-8) == rank(Matrix(S), rtol = 1e-8)
end

function test_S_row_space(S)
    for i in 1:16
        ps = Pm[:, i]
        print("$(ps_labels[i]) :")
        if test_contains(S, ps)
            println("True")
        else
            println("False")
        end
    end
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

## ==================== Parameters and system generation ========================
ϵ_func() = 0.5
ϵb_func() = [0, 0, 1]
u_intra_func() = rand() + 10
t_func() = rand()
t_so_func() = 0
u_inter_func() = rand()

nbr_dots_res = 2
qn_res = 1
sys = tight_binding_system(2, nbr_dots_res, qn_res)

hams = QDR.matrix_representation_hams(
    get_ham(sys.grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func),
    sys)

measurements = QDR.charge_probabilities(sys)

Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)

t_list = range(0, 20, 100)
S_list = map(
    t -> scrambling_map(sys, measurements, ground_state(hams.res), hams.total, t), t_list)

S_pauli_list = [S * Pm .* (1 / 4) for S in S_list]
S_pauli_overlaps = vcat([sum(abs2.(S_p), dims = 1) for S_p in S_pauli_list]...)

## Plotting and tests
plot_Pauli_overlap_over_time(t_list, S_pauli_overlaps, Pm_dict)
test_S_row_space(S_list[end])