using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions
import QDReservoir as QDR
## ======================= Functions =============================
function plot_mse_weights(nbr_states_list, mse_matrix, W_list)
    fig = Figure(size = (600, 300))
    ax1 = Axis(fig[1, 1], xlabel = "Number of training states",
        ylabel = "Mean Squared Error", title = "MSE vs Training Data", yscale = log10,
        xticks = [0, 16, 20, 40, 60, 80, 100])
    mse_mtrix_avg = vec(mean(mse_matrix, dims = 2))
    lines!(ax1, nbr_states_list, mse_mtrix_avg, label = "Average MSE of\nspin predictions")
    lines!(ax1, nbr_states_list, W_list, label = "||W - R|| / ||R||")
    vlines!(
        ax1, [16], color = :grey, linestyle = :dash)
    axislegend(ax1, position = :rt)
    display(fig)
end

function plot_mse_weights_compare(
        nbr_states_list1, nbr_states_list2, mse_matrix1, mse_matrix2, W_list1, W_list2)
    fig = Figure(size = (600, 250))
    ax1 = Axis(fig[1, 1], xlabel = "Number of training states",
        ylabel = "Mean Squared Error", title = "Noise Free Measurements", yscale = log10,
        xticks = [0, 16, 50, 100])
    mse_mtrix_avg1 = vec(mean(mse_matrix1, dims = 2))
    mse_mtrix_avg2 = vec(mean(mse_matrix2, dims = 2))
    lines!(
        ax1, nbr_states_list1, mse_mtrix_avg1, label = "Average Spin Estimation")
    lines!(
        ax1, nbr_states_list1, W_list1, label = "Weight Matrix")
    vlines!(ax1, [16], color = :grey, linestyle = :dash)

    ax2 = Axis(fig[1, 2], xlabel = "Number of training states",
        ylabel = "Mean Squared Error", title = "Noisy Measurements, σE: $(σE)", yscale = log10)
    lines!(ax2, nbr_states_list2, mse_mtrix_avg2,
        label = "Average Spin Estimation")
    lines!(
        ax2, nbr_states_list2, W_list2, label = "Weight Matrix and Recovery Map Difference")
    Legend(fig[2, 1:2], ax1,
        orientation = :horizontal, framevisible = false)
    return fig
end
