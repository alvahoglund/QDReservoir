using QDReservoir, LinearAlgebra, CairoMakie, Random
import QDReservoir as QDR

nbr_dots_res = 3
qn_res = 1
sys = tight_binding_system(2, nbr_dots_res, qn_res)
param = QDR.random_param_functions()
seed = 1234
Random.seed!(seed)
hams = QDR.matrix_representation_hams(
    QDR.hamiltonians(sys.grids, param),
    sys)
ψ_res = QDR.ground_state(hams.res)

t_range = range(10^(-8), 50, 1000)
op = QDR.p1(sys.grids.res[end])
op_m = matrix_representation(op, sys.H_total)

S_list = [scrambling_map(sys, [op_m], ψ_res, hams.total, t) for t in t_range]
S = reduce(vcat, S_list)

Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
pauli_overlaps = abs2.(S * Pm .* (1 / 4))
clean_val(y) = map(x -> abs(x) < 1e-10 ? 0.0 : x, y)

#chosen_pauli_strings = [(:σx, :σ0), (:σy, :σ0), (:σz, :σ0), (:σ0, :σx), (:σx, :σx), (:σy, :σx), (:σz, :σx),
#    (:σ0, :σy), (:σx, :σy), (:σy, :σy), (:σz, :σy), (:σ0, :σz), (:σx, :σz),
#    (:σy, :σz), (:σz, :σz)]

plot_pauli_strings = [(:σ0, :σz), (:σx, :σy), (:σx, :σx), (:σz, :σz)]
fig = Figure(size = (600, 400))
ax1 = Axis(fig[1, 1], title = "Pauli string coefficients of effective measurements",
    xlabel = "time",
    ylabel = "Overlap |c|²", xgridvisible = false, ygridvisible = false)
ylims!(ax1, low = 0, high = 1)
#Plot (:σ0, :σ0)
idx = Pm_dict[(:σ0, :σ0)]
lines!(ax1, t_range, clean_val.(pauli_overlaps[:, idx]); label = "σ0 ⊗ σ0")
axislegend(ax1, framevisible = false, position = :lb)

ax2 = Axis(fig[2, 1], xlabel = "time",
    ylabel = "Overlap |c|²", xgridvisible = false, ygridvisible = false)
ylims!(ax2, low = 0)
for ps in plot_pauli_strings
    idx = Pm_dict[ps]
    lines!(ax2, t_range, clean_val.(pauli_overlaps[:, idx]); label = "$(ps[1]) ⊗ $(ps[2])")
end
axislegend(ax2, framevisible = false, position = :lt)
display(fig)
save("Figures/effective_measurement_dynamics.png", fig)