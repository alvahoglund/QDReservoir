using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions, JLD2, Random
import QDReservoir as QDR

includet("estimate_spin.jl")
includet("plots_estimate_spin.jl")

## ============== Generate/ load data ===================
ham_params = QDR.ParamFunctions(
    ϵ_func_main = () -> 0.5,
    ϵ_func_res = () -> 0.5,
    ϵb_func = () -> [0, 0, 1],
    u_intra_func = () -> rand() + 10,
    t_func = () -> rand(),
    t_so_func = () -> 0.1 * rand(),
    u_inter_func = () -> rand())
nbr_dots_res = 6
qn_res = 3

sys = tight_binding_system(2, nbr_dots_res, qn_res)
seed = 1234
Random.seed!(seed)
hams = QDR.matrix_representation_hams(QDR.hamiltonians(sys.grids, ham_params), sys)
t = [100, 200]

measurements = map(m -> matrix_representation(m, sys.H_total),
    QDR.charge_probabilities(sys.grids.total))
S = scrambling_map(sys, measurements, ground_state(hams.res),
    hams.total, t)
# S = load("DefaultSystems/scrambling_map_A.jld2", "S")
# sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")

Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
nbr_states = 10^4
Ω = randomize_hs_states(sys, nbr_states)
X = QDR.process_complex.((S * Ω)')
Y = QDR.process_complex.((Pm' * Ω)')

## ================= Train model and evaluate ==================
σE = 1e-6
result = train_model(X, Y, σE)
mse_per_pauli = vec(mean((result.Y_test - result.Y_pred) .^ 2, dims = 1))

nbr_train = size(X, 1) ÷ 2
Y_train = Y[1:nbr_train, :]

## ================= Figures ======================
plot_paulis = [(:σz, :σz)]

fig1 = plot_training_vs_test(
    plot_paulis, result.Y_test, result.Y_pred, Pm_dict, mse_per_pauli, σE)
#save("Figures/training_vs_test.png", fig1)

fig2 = plot_test_and_prediction(
    plot_paulis, result.Y_test, result.Y_pred, Pm_dict, mse_per_pauli)
#save("Figures/test_and_prediction.png", fig2)

fig3 = plot_test(plot_paulis, result.Y_test, Pm_dict)
#save("Figures/test_labels.png", fig3)

fig4 = plot_training(plot_paulis, Y_train, Pm_dict)
#save("Figures/training_labels.png", fig4)
