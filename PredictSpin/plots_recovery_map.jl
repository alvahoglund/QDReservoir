using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions
import QDReservoir as QDR

function plot_mse_weights!(ax, nbr_states_list, mse_matrix, W_list)
    mse_avg = vec(mean(mse_matrix, dims = 2))
    lines!(ax, nbr_states_list, mse_avg, label = "Average MSE of\nspin predictions")
    lines!(ax, nbr_states_list, W_list, label = "||W - R|| / ||R||")
    vlines!(ax, [16], color = :grey, linestyle = :dash)
    axislegend(ax, position = :rt)
end

function plot_mse_weights(nbr_states_list, mse_matrix, W_list)
    fig = Figure(size = (600, 300))
    ax = Axis(fig[1, 1], xlabel = "Number of training states",
        ylabel = "MSE", title = "MSE vs Training Data",
        yscale = log10, xticks = [0, 16, 20, 40, 60, 80, 100],
        xgridvisible = false, ygridvisible = false)
    plot_mse_weights!(ax, nbr_states_list, mse_matrix, W_list)
    return fig
end

function plot_mse_weights_compare!(gl, nbr_states_list1, nbr_states_list2,
        mse_matrix1, mse_matrix2, W_list1, W_list2, σE)
    mse_avg1 = vec(mean(mse_matrix1, dims = 2))
    mse_avg2 = vec(mean(mse_matrix2, dims = 2))

    ax1 = Axis(gl[1, 1], xlabel = "Number of training states",
        ylabel = "MSE", title = "Noise Free Measurements",
        yscale = log10, xticks = [0, 16, 50, 100],
        xgridvisible = false, ygridvisible = false)
    lines!(ax1, nbr_states_list1, mse_avg1, label = "Spin prediction")
    lines!(ax1, nbr_states_list1, W_list1, label = "Weight matrix")
    vlines!(ax1, [16], color = :grey, linestyle = :dash)

    ax2 = Axis(gl[1, 2], xlabel = "Number of training states",
        ylabel = "MSE", title = "Noisy Measurements, σE = $σE",
        yscale = log10, xgridvisible = false, ygridvisible = false)
    lines!(ax2, nbr_states_list2, mse_avg2, label = "Spin prediction")
    lines!(ax2, nbr_states_list2, W_list2, label = "Weight matrix")

    axislegend(ax1, position = :rt, framevisible = false)
end

function plot_mse_weights_compare(nbr_states_list1, nbr_states_list2,
        mse_matrix1, mse_matrix2, W_list1, W_list2, σE)
    fig = Figure(size = (600, 250))
    plot_mse_weights_compare!(fig.layout, nbr_states_list1, nbr_states_list2,
        mse_matrix1, mse_matrix2, W_list1, W_list2, σE)
    return fig
end
