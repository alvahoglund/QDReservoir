using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions
import QDReservoir as QDR
## ======================= Functions =============================
function randomize_states(sys, nbr_states)
    stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main)) for i in 1:nbr_states)
end

function add_noise(X, σE)
    E = rand(Normal(0, σE), size(X))
    return X + E
end

function train_model(Ω_train, S, Pm, σE)
    X_train = QDR.process_complex.((S * Ω_train)')
    E_train = rand(Normal(0, σE), size(X_train))

    X̃_train = X_train + E_train
    Y_train = QDR.process_complex.((Pm' * Ω_train)')

    W = pinv(X̃_train) * Y_train
    return W
end

function test_model(W, X_test, Y_true)
    Y_pred = X_test * W
    mse = mean((Y_true - Y_pred) .^ 2, dims = 1)
    return mse
end

function get_A(S, σE)
    SVD_S = svd(S)
    b = 0.0147
    return SVD_S.U * diagm(b .* SVD_S.S .^ 2 ./ (b .* SVD_S.S .^ 2 .+ σE^2)) * SVD_S.U'
end

function test_weights(W, Pm, S, σE)
    A = get_A(S, σE)
    R = A * pinv(S') * Pm
    return norm(W - R, 2) / norm(R, 2)
end

function evaluate_model(sys, X_test, S, Pm, σE, nbr_states_train)
    Ω_train = randomize_states(sys, nbr_states_train)
    W = train_model(Ω_train, S, Pm, σE)
    Y_true = QDR.process_complex.((Pm' * Ω_test)')
    mse = test_model(W, X_test, Y_true)
    weight_error = test_weights(W, Pm, S, σE)
    return mse, weight_error
end

function vary_training_data(sys, X_test, S, Pm, σE, nbr_states_list)
    map(nbr_states_train -> evaluate_model(sys, X_test, S, Pm, σE, nbr_states_train),
        nbr_states_list)
end

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
    fig = Figure(size = (800, 300))
    ax1 = Axis(fig[1, 1], xlabel = "Number of training states",
        ylabel = "Mean Squared Error", title = "Noise Free Measurements", yscale = log10,
        xticks = [0, 16, 50, 100])
    mse_mtrix_avg1 = vec(mean(mse_matrix1, dims = 2))
    mse_mtrix_avg2 = vec(mean(mse_matrix2, dims = 2))
    lines!(
        ax1, nbr_states_list1, mse_mtrix_avg1, label = "Spin prediction")
    lines!(ax1, nbr_states_list1, W_list1, label = "Weight matrix")
    vlines!(ax1, [16], color = :grey, linestyle = :dash)
    axislegend(ax1, position = :rt)

    ax2 = Axis(fig[1, 2], xlabel = "Number of training states",
        ylabel = "Mean Squared Error", title = "Noisy Measurements, σE: $(σE)", yscale = log10)
    lines!(ax2, nbr_states_list2, mse_mtrix_avg2,
        label = "Spin prediction")
    lines!(ax2, nbr_states_list2, W_list2, label = "Weight matrix")
    axislegend(ax2, position = :rt)
    return fig
end

##
nbr_dots_res = 6
qn_res = 3
sys = tight_binding_system(2, nbr_dots_res, qn_res)
seed = 1323
hams = QDR.matrix_representation_hams(QDR.hamiltonians(sys.grids, seed), sys)
ρ_res = ground_state(hams.res)
t_list = [100, 200]
σE = 10^-4
measurements = QDR.charge_measurements(sys)
Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
S = scrambling_map(sys, measurements, ρ_res, hams.total, t_list)

nbr_states_test = 10^5
Ω_test = randomize_states(sys, nbr_states_test)
X_test = QDR.process_complex.((S * Ω_test)')
X̃_test = add_noise(X_test, σE)

##
nbr_states_list1 = [i for i in range(1, 100, 100)]
nbr_states_list2 = [i for i in range(1, 10^4, 100)]
model_result = vary_training_data(sys, X_test, S, Pm, 0, nbr_states_list1)
model_result_noisy = vary_training_data(sys, X̃_test, S, Pm, σE, nbr_states_list2)

mse_matrix = vcat(getindex.(model_result, 1)...)
W_list = getindex.(model_result, 2)

plot_mse_weights(nbr_states_list1, mse_matrix, W_list)
mse_matrix_noisy = vcat(getindex.(model_result_noisy, 1)...)
W_list_noisy = getindex.(model_result_noisy, 2)
fig1 = plot_mse_weights_compare(
    nbr_states_list1, nbr_states_list2, mse_matrix, mse_matrix_noisy, W_list, W_list_noisy)
#Save the figure
save("Figures/mse_weights_comparison.png", fig1)