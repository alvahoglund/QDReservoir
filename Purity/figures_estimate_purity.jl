includet("estimate_purity.jl")
includet("plots_estimate_purity.jl")
using JLD2

function predict_state(X, Y, σE)
    X_train, X_test, Y_train, Y_test = QDR.split_train_test(X, Y)
    X̃_test = QDR.add_noise(X_test, σE)
    X̃_train = QDR.add_noise(X_train, σE)
    W, Y_pred = QDR.ridge_regression(X̃_train, Y_train, X̃_test)
    return Y_pred, Y_test
end

function get_purity_mse_from_predicted_states(X, Ω, σE)
    Y_pred, Y_test = predict_state(X, Ω', σE)
    return QDR.mse(get_purity(Y_test'), get_purity(Y_pred'))
end

##
S = load("DefaultSystems/scrambling_map_A.jld2", "S")
sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")

Ω = random_mixed_states(10^5, sys)
X = QDR.process_complex.((S * Ω)')
Y = get_purity(Ω)

## ============ Example of predicting purity ===================== 
σE = 10^-2
λ = 0

W, Y_pred, Y_test = estimate_purity(X, Y, σE)

Ω_pred, Ω_test = predict_state(X, Ω', σE)

mse1 = mean((Y_pred - Y_test) .^ 2)
mse2 = mean((get_purity(Ω_pred') - get_purity(Ω_test')) .^ 2)

println("MSE of purity prediction: $mse1")
println("MSE of purity prediction from predicted states: $mse2")

##
plot_test_vs_pred_purity(Y_test, Y_pred)
## ========== Plot purity MSE against noise levels ==========

σE_list = 10 .^ range(-5, 0, length = 20)
mse_list = vcat([get_purity_mse(X, Y, σE) for σE in σE_list]...)
mse_list_state_pred = vcat([get_purity_mse_from_predicted_states(X, Ω, σE)
                            for σE in σE_list]...)

fig = Figure(size = (600, 250))
ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "Purity MSE",
    title = "Purity prediction performance", xscale = log10)
lines!(ax, σE_list, mse_list, label = "Purity MSE")
lines!(ax, σE_list, mse_list_state_pred, label = "Purity MSE from predicted states")
axislegend(position = :lt)
fig

plot_purity_mse(σE_list, mse_list)
plot_purity_mse(σE_list, mse_list_state_pred)