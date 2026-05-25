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

function test_weights(W, Pm, S)
    R = pinv(S') * Pm
    return norm(W - R)
end

function evaluate_model(sys, X_test, S, Pm, σE, nbr_states_train)
    Ω_train = randomize_states(sys, nbr_states_train)
    W = train_model(Ω_train, S, Pm, σE)
    Y_true = QDR.process_complex.((Pm' * Ω_test)')
    mse = test_model(W, X_test, Y_true)
    weight_error = test_weights(W, Pm, S)
    return mse, weight_error
end

function vary_training_data(sys, X_test, S, Pm, σE, nbr_states_list)
    map(nbr_states_train -> evaluate_model(sys, X_test, S, Pm, σE, nbr_states_train),
        nbr_states_list)
end

function plot_mse_weights(nbr_states_list, mse_matrix, W_list, Pm_dict)
    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel = "Number of training states",
        ylabel = "Mean Squared Error", title = "MSE vs Training Data", yscale = log10)

    lines!(ax1, nbr_states_list, mse_matrix[:, Pm_dict[(:σz, :σz)]], label = "σz ⊗ σz")
    lines!(ax1, nbr_states_list, mse_matrix[:, Pm_dict[(:σx, :σy)]], label = "σx ⊗ σy")
    axislegend(ax1, position = :rt)

    ax2 = Axis(fig[2, 1], xlabel = "Number of training states",
        ylabel = "Weight difference", title = "||W - R|| vs Training Data", yscale = log10)
    lines!(ax2, nbr_states_list, W_list)
    display(fig)
end

##
nbr_dots_res = 6
qn_res = 3
sys = tight_binding_system(2, nbr_dots_res, qn_res)
seed = 1234
hams = QDR.matrix_representation_hams(QDR.hamiltonians(sys.grids, seed), sys)
ρ_res = ground_state(hams.res)
t_list = [100]
σE = 0
measurements = QDR.charge_measurements(sys)
Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
S = scrambling_map(sys, measurements, ρ_res, hams.total, t_list)

nbr_states_test = 10^5
Ω_test = randomize_states(sys, nbr_states_test)
X_test = QDR.process_complex.((S * Ω_test)')
X̃_test = add_noise(X_test, σE)

##
nbr_states_list = [i for i in range(1, 50)]
model_result = vary_training_data(sys, X̃_test, S, Pm, σE, nbr_states_list)

mse_matrix = vcat(getindex.(model_result, 1)...)
W_list = getindex.(model_result, 2)

plot_mse_weights(nbr_states_list, mse_matrix, W_list, Pm_dict)
