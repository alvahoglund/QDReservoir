includet("estimate_purity.jl")
includet("plots_estimate_purity.jl")
using JLD2
##
S = load("DefaultSystems/scrambling_map_A.jld2", "S")
sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")

Ω = random_mixed_states(10^5, sys)
X = QDR.process_complex.((S * Ω)')
Y = get_purity(Ω)

## ============ Example of predicting purity ===================== 
σE = 10^-5
λ = 0

W, Y_pred, Y_test = estimate_purity(X, Y, σE)
plot_test_vs_pred_purity(Y_test, Y_pred)

## ========== Plot purity MSE against noise levels ==========

σE_list = 10 .^ range(-5, 0, length = 20)
mse_list = vcat([get_purity_mse(X, Y, σE) for σE in σE_list]...)
plot_purity_mse(σE_list, mse_list)
