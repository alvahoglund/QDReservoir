using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions, JLD2
import QDReservoir as QDR

includet("estimate_spin.jl")
includet("plots_estimate_spin_noise.jl")

## ================= Load system ======================
S = load("DefaultSystems/scrambling_map_A.jld2", "S")
sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")
Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)

## ================= Generate states ======================
nbr_states_small = 10^3
nbr_states_large = 10^5

Ω_small = randomize_hs_states(sys, nbr_states_small)
Ω_large = randomize_hs_states(sys, nbr_states_large)

X_small = QDR.process_complex.((S * Ω_small)')
Y_small = QDR.process_complex.((Pm' * Ω_small)')
X_large = QDR.process_complex.((S * Ω_large)')
Y_large = QDR.process_complex.((Pm' * Ω_large)')

S_SVD = svd(S)
sv_overlaps = SV_overlap(S_SVD, Pm)
b = 0.0147

## ===== PLOT VARYING NOISE ===========
ps_list = [(:σx, :σz), (:σ0, :σx)]
σE_list = 10 .^ range(-7, 0, length = 30)

mse_list = vcat([get_mse(X_large, Y_large, σE)' for σE in σE_list]...)
mse_pred_list = vcat([mse_prediction(S_SVD, Pm, σE, b)' for σE in σE_list]...)

fig1 = plot_varying_noise(
    σE_list, mse_list, mse_pred_list, S_SVD, sv_overlaps, Pm_dict, ps_list, b)
#save("Figures/varying_noise_spin_estimation.png", fig1)

## ===== PLOT MODE DECOMPOSITION ===========
σE_list = 10 .^ range(-7, 0, length = 200)

mse_mat_small = vcat([get_mse(X_small, Y_small, σE)' for σE in σE_list]...)
mse_mat_large = vcat([get_mse(X_large, Y_large, σE)' for σE in σE_list]...)

fig2 = plot_mode_decomposition(
    σE_list, mse_mat_small, mse_mat_large, S_SVD, Pm, Pm_dict, (:σy, :σx), b)
#save("Figures/mode_decomposition_σy_σx.png", fig2)
