##
using QDReservoir
using LinearAlgebra, Statistics, GLMakie, Distributions, Random
import QDReservoir as QDR

## 
function default_system()
    nbr_dots_res = 6
    qn_res = 3
    sys = tight_binding_system(2, nbr_dots_res, qn_res)
    hams = QDR.matrix_representation_hams(hamiltonians(sys.grids), sys)
    return sys, hams
end

function default_scrambling(sys, hams)
    t = 100
    measurements = map(m -> matrix_representation(m, sys.H_total),
        QDR.charge_probabilities(sys.grids.total))
    return scrambling_map(sys, measurements, ground_state(hams.res),
        hams.total, t)
end

function random_mixed_states(nbr_states, sys)
    p_list = rand(nbr_states)
    mapreduce(
        i -> vec((1 - p_list[i]) * density_matrix(QDR.random_state(sys.H_main)) +
                 p_list[i] * QDR.max_mixed_state(sys.H_main)),
        hcat, 1:nbr_states)
end

function add_noise(σE, X)
    E = rand(Normal(0, σE), size(X))
    return X + E
end

function plot_test_vs_pred_purity(Y_test, Y_pred)
    #Sort first on Y_test, then on Y_pred
    sort_indices = sortperm(Y_pred)
    Y_test = Y_test[sort_indices]
    Y_pred = Y_pred[sort_indices]
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "True labels", ylabel = "Predicted labels",
        title = "Ridge regression for purity estimation, MSE = $(round(mse(Y_test, Y_pred), digits = 5))")
    scatter!(ax, Y_test, Y_pred, label = "Predicted vs True")
    lines!(ax, [0, 1], [0, 1], linestyle = :dash,
        color = :red, label = "Perfect predictions")
    axislegend(position = :lt)
    display(fig)
end

get_purity(Ω) = [real(dot(Ω[:, i], Ω[:, i])) for i in eachindex(Ω[1, :])]

function predict_purity(X, Y, σE)
    X̃ = add_noise(σE, X)
    nbr_states = size(X, 1)
    nbr_train = nbr_states ÷ 2
    X̃_train, X̃_test = X̃[1:nbr_train, :], X̃[(nbr_train + 1):end, :]
    feature_transformation_func = degree_2_polynomial_feature_transformation
    X̃_train_poly = feature_transformation_func(X̃_train)
    X̃_test_poly = feature_transformation_func(X̃_test)
    Y_train, Y_test = Y[1:nbr_train], Y[(nbr_train + 1):end]
    W, Y_pred = ridge_regression(X̃_train_poly, Y_train, X̃_test_poly, λ)
    return W, Y_pred, Y_test
end

function get_purity_mse(X, Y, σE)
    W, Y_pred, Y_true = predict_purity(X, Y, σE)
    return mse(Y_true, Y_pred)
end

function plot_purity_mse(σE_list, mse_list, vlines_list = nothing)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "Mean Squared Error",
        title = "MSE of Ridge regression for purity estimation", xscale = log10)
    lines!(ax, σE_list, mse_list, label = "MSE")
    if vlines_list !== nothing
        vlines!(ax, vlines_list, linestyle = :dash, color = :grey, label = "Examples")
    end
    axislegend(position = :lt)
    display(fig)
end

mse_pred(X_S, X_U, Y, σ_E) = Y' * X_U * diagm(σ_E^2 ./ ((X_S .^ 2) .+ σ_E^2)) * X_U' * Y
function predict_mse(X, σE_list)
    X_U, X_S, X_V = svd(X)
    mse_pred_list = vcat([mse_pred(X_S, X_U, Y, σE) for σE in σE_list]...)
    return mse_pred_list
end

function plot_mse_and_pred_mse(σE_list, mse_list, mse_pred_list)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "Mean Squared Error",
        title = "MSE of Ridge regression for purity estimation", xscale = log10)
    lines!(ax, σE_list, mse_list, label = "MSE")
    lines!(ax, σE_list, mse_pred_list, label = "Predicted MSE")
    axislegend(position = :lt)
    display(fig)
end
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