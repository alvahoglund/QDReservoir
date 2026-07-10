includet("estimate_purity.jl")
includet("plots_estimate_purity.jl")
using JLD2
##
S = load("DefaultSystems/scrambling_map_A.jld2", "S")
sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")

Ω = random_mixed_states(10^5, sys.H_main)
X = QDR.process_complex.((S * Ω)')
Y = get_purity(Ω)

## ============ Example of predicting purity ===================== 
σE = 10^-5
λ = 0

W, Y_pred, Y_test = estimate_purity(X, Y, σE)
Y_pred_state, Y_test_state = purity_from_state_estimation(X, Ω, σE)

plot_test_vs_pred_purity(Y_test, Y_pred)
plot_test_vs_pred_purity(reshape(Y_test_state, :, 1), reshape(Y_pred_state, :, 1))

## ========== Plot purity MSE against noise levels ==========

σE_list = 10 .^ range(-5, 0, length = 20)
mse_list = vcat([get_purity_mse(X, Y, σE) for σE in σE_list]...)
mse_list_from_state = vcat([purity_from_state_estimation_mse(X, Ω, σE) for σE in σE_list]...)

fig = Figure(size = (600, 300))
ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "MSE",
    title = "Purity Estimation MSE vs Noise Level", xscale = log10)
lines!(ax, σE_list, mse_list, label = "Feature transformation")
lines!(ax, σE_list, mse_list_from_state, label = "State Estimation")
axislegend(ax, position = :lt)
save("Figures/estimate_purity_methods.png", fig)