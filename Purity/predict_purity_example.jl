includet("predict_purity.jl")
## ============ Define system ======================
seed = 1238
Random.seed!(seed)
sys, hams = default_system()
S = default_scrambling(sys, hams)
nbr_states = 10^5
Ω = random_mixed_states(nbr_states, sys)
X = QDR.process_complex.((S * Ω)')
Y = get_purity(Ω)

## ============ Example of predicting purity ===================== 
σE = 0
λ = 0

W, Y_pred, Y_test = predict_purity(X, Y, σE)
plot_test_vs_pred_purity(Y_test, Y_pred)
mse(Y_test, Y_pred)

## ========== Plot purity MSE against noise levels ==========

σE_list = 10 .^ range(-10, 0, length = 50)
mse_list = vcat([get_purity_mse(X, Y, σE) for σE in σE_list]...)
plot_purity_mse(σE_list, mse_list, [10^-10, 10^-4])

## ========== Plot predicted MSE against noise levels ==========
c = 20
mse_pred_list = predict_mse(X, σE_list)
plot_mse_and_pred_mse(σE_list, mse_list, mse_pred_list ./ c)