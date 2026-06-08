using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions, JLD2
import QDReservoir as QDR

includet("estimate_spin.jl")
includet("plots_recovery_map.jl")

## ================= Load system ======================
S = load("DefaultSystems/scrambling_map_A.jld2", "S")
sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")
Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)

## ================= Parameters ======================
σE = 1e-3
b = 0.0147

## ================= Generate test states ======================
nbr_states_test = 10^5
Ω_test = randomize_hs_states(sys, nbr_states_test)
X_test = QDR.process_complex.((S * Ω_test)')
Y_test = QDR.process_complex.((Pm' * Ω_test)')
X̃_test = QDR.add_noise(X_test, σE)

## 
# Vary the number of traning states and evaluate the MSE and weight convergence 
nbr_states_list1 = [i for i in range(1, 100, 100)]
nbr_states_list2 = [i for i in range(1, 10^4, 100)]
model_result = vary_training_data(
    sys, X_test, Y_test, S, Pm, 0.0, b, nbr_states_list1)
model_result_noisy = vary_training_data(
    sys, X̃_test, Y_test, S, Pm, σE, b, nbr_states_list2)

mse_matrix = vcat([r.mse' for r in model_result]...)
W_list = [r.weight_err for r in model_result]

plot_mse_weights(nbr_states_list1, mse_matrix, W_list)
mse_matrix_noisy = vcat([r.mse' for r in model_result_noisy]...)
W_list_noisy = [r.weight_err for r in model_result_noisy]
fig1 = plot_mse_weights_compare(
    nbr_states_list1, nbr_states_list2, mse_matrix, mse_matrix_noisy, W_list, W_list_noisy, σE)
#Save the figure
#save("Figures/mse_weights_comparison.png", fig1)