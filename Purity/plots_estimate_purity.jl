using QDReservoir
using LinearAlgebra, Statistics, Distributions, Random, CairoMakie
import QDReservoir as QDR
##

function plot_test_vs_pred_purity(Y_test, Y_pred)
    #Sort first on Y_test, then on Y_pred
    sort_indices = sortperm(Y_pred, dims = 1)
    Y_test = Y_test[sort_indices]
    Y_pred = Y_pred[sort_indices]
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "True labels", ylabel = "Predicted labels",
        title = "Ridge regression for purity estimation, MSE = $(round(QDR.mse(Y_test, Y_pred), digits = 5))")
    scatter!(ax, vec(Y_test), vec(Y_pred), label = "Predicted vs True")
    lines!(ax, [0, 1], [0, 1], linestyle = :dash,
        color = :red, label = "Perfect predictions")
    axislegend(position = :lt)
    display(fig)
end

function plot_purity_mse(σE_list, mse_list, vlines_list = nothing)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "MSE",
        title = "MSE of Ridge regression for purity estimation", xscale = log10)
    lines!(ax, σE_list, mse_list, label = "MSE")
    if vlines_list !== nothing
        vlines!(ax, vlines_list, linestyle = :dash, color = :grey, label = "Examples")
    end
    axislegend(position = :lt)
    display(fig)
end
