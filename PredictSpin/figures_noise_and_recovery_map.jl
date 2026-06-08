using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions, JLD2
import QDReservoir as QDR

includet("estimate_spin.jl")
includet("plots_estimate_spin_noise.jl")
includet("plots_recovery_map.jl")

## ================= Load system ======================
S = load("DefaultSystems/scrambling_map_A.jld2", "S")
sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")
Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
S_SVD = svd(S)

## ================= Parameters ======================
σE = 1e-3
b = 0.0147

## ================= Data for varying noise ======================
Ω_large = randomize_hs_states(sys, 10^5)
X_large = QDR.process_complex.((S * Ω_large)')
Y_large = QDR.process_complex.((Pm' * Ω_large)')
sv_overlaps = SV_overlap(S_SVD, Pm)

ps_list = [(:σx, :σz), (:σ0, :σx)]
σE_list = 10 .^ range(-7, 0, length = 30)
mse_list = vcat([get_mse(X_large, Y_large, σE)' for σE in σE_list]...)
mse_pred_list = vcat([mse_prediction(S_SVD, Pm, σE, b)' for σE in σE_list]...)

## ================= Data for recovery map ======================
Ω_test = randomize_hs_states(sys, 10^5)
X_test = QDR.process_complex.((S * Ω_test)')
Y_test = QDR.process_complex.((Pm' * Ω_test)')
X̃_test = QDR.add_noise(X_test, σE)

nbr_states_list1 = round.(Int, range(1, 100, length = 100))
nbr_states_list2 = round.(Int, range(1, 10^4, length = 100))

results_noisefree = vary_training_data(sys, X_test, Y_test, S, Pm, 0.0, b, nbr_states_list1)
results_noisy = vary_training_data(sys, X̃_test, Y_test, S, Pm, σE, b, nbr_states_list2)

mse_matrix1 = vcat([r.mse' for r in results_noisefree]...)
W_list1 = [r.weight_err for r in results_noisefree]
mse_matrix2 = vcat([r.mse' for r in results_noisy]...)
W_list2 = [r.weight_err for r in results_noisy]

## ================= Combined figure ======================
fig = Figure(size = (600, 450))

gl_top = fig[1, 1] = GridLayout()
gl_bottom = fig[2, 1] = GridLayout()

plot_mse_weights_compare!(gl_top, nbr_states_list1, nbr_states_list2,
    mse_matrix1, mse_matrix2, W_list1, W_list2, σE)
plot_varying_noise!(
    gl_bottom, σE_list, mse_list, mse_pred_list, S_SVD, sv_overlaps, Pm_dict, ps_list, b)
rowgap!(fig.layout, 1, 20)

fig
#save("Figures/spin_overview.png", fig)
