using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions
import QDReservoir as QDR
##  ======================= Plotting Functions =============================
function plot_training_vs_test(plot_paulis, Y_test, Y_pred, Pm_dict, mean_squared_error, σE)
    x_vals = range(-1, 1, length = 100)
    fig = Figure(fontsize = 15)
    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        ax = Axis(fig[1, i],
            xlabel = "True value",
            ylabel = "Predicted value",
            title = "Measurement: $(ps[1]) ⊗ $(ps[2]), MSE: $(round(mean_squared_error[idx], digits=4)), σE: $(round(σE, digits=4))")
        scatter!(ax, Y_test[:, idx], Y_pred[:, idx],
            label = L"Y_{test} vs Y_{pred}", color = :orange, markersize = 15)
        lines!(ax, x_vals, x_vals, color = :black, label = "Optimal")
        axislegend(position = :rb)
    end

    display(fig)
end

function plot_training(plot_paulis, Y_train, Pm_dict)
    fig_train = Figure(fontsize = 15)

    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        x_range = range(1, length(Y_train[:, idx]))
        ax = Axis(fig_train[1, i],
            ylabel = "Spin expectation value",
            title = "Measurement: $(ps[1]) ⊗ $(ps[2])")
        sort_idx = sortperm(Y_train[:, idx])
        scatter!(ax, x_range, Y_train[:, idx][sort_idx],
            label = L"Y_{train}", color = :orange, markersize = 15)
        axislegend(position = :rb)
    end
    display(fig_train)
end

function plot_test(plot_paulis, Y_test, Pm_dict)
    fig_test = Figure(fontsize = 15)

    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        x_range = range(1, length(Y_test[:, idx]))
        ax = Axis(fig_test[1, i],
            ylabel = "Spin expectation value",
            title = "Measurement: $(ps[1]) ⊗ $(ps[2])")
        sort_idx = sortperm(Y_test[:, idx])
        scatter!(ax, x_range, Y_test[:, idx][sort_idx],
            label = L"Y_{test}", color = :orange, markersize = 15)
        axislegend(position = :rb)
    end
    display(fig_test)
end

function plot_test_and_prediction(plot_paulis, Y_test, Y_pred, Pm_dict, mean_squared_error)
    fig_test_pred = Figure(fontsize = 15)

    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        x_range = range(1, length(Y_test[:, idx]))
        ax = Axis(fig_test_pred[1, i],
            ylabel = "Spin expectation value",
            title = "Measurement: $(ps[1]) ⊗ $(ps[2]), MSE: $(round(mean_squared_error[idx], digits=4))")
        sort_idx = sortperm(Y_test[:, idx])
        scatter!(ax, x_range, Y_test[:, idx][sort_idx],
            label = L"Y_{test}", color = :orange, markersize = 15)
        scatter!(ax, x_range, Y_pred[:, idx][sort_idx],
            label = L"Y_{pred}", marker = :cross, color = :black, markersize = 12)
        axislegend(position = :rb)
    end
    display(fig_test_pred)
end

ps_labels = [("$(a) ⊗ $(b)")
             for a in ["σ0", "σx", "σy", "σz"], b in ["σ0", "σx", "σy", "σz"]]

function test_contains(S, ps)
    rank(Matrix(vcat(S, ps')), rtol = 1e-8) == rank(Matrix(S), rtol = 1e-8)
end

function test_S_row_space(S)
    for i in 1:16
        ps = Pm[:, i]
        print("$(ps_labels[i]) :")
        if test_contains(S, ps)
            println("True")
        else
            println("False")
        end
    end
end
## ================= Functions for generating  data ======================
function get_ham(grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.main)
    reservoir_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.res)
    interaction_parameters = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(grids, main_system_parameters,
        reservoir_parameters, interaction_parameters)
end
## ================= Parameters for system generation ======================

ϵ_func() = 0.5
ϵb_func() = [0, 0, 1]
u_intra_func() = rand() + 10
t_func() = rand()
t_so_func() = 0.1 * rand()
u_inter_func() = rand()

nbr_dots_res = 6
qn_res = 3
sys = tight_binding_system(2, nbr_dots_res, qn_res)

#hams = QDR.matrix_representation_hams(
#    get_ham(sys.grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func),
#    sys)
seed = 1234
hams = QDR.matrix_representation_hams(QDR.hamiltonians(sys.grids, seed), sys)
nbr_states = 1000
nbr_train = nbr_states ÷ 2
nbr_test = nbr_states - nbr_train
σE = 0
t = 100

# Charge Measurements, X 
measurements = map(m -> matrix_representation(m, sys.H_total),
    QDR.charge_probabilities(sys.grids.total))
Ω = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main)) for i in 1:nbr_states)
S = scrambling_map(sys, measurements, ground_state(hams.res),
    hams.total, t)

X = QDR.process_complex.((S * Ω)')
E = rand(Normal(0, σE), size(X))
X̃ = X + E

X_train, X_test = X̃[1:nbr_train, :], X̃[(nbr_train + 1):nbr_states, :]

# Spin Measurements, Y 
Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
Y = QDR.process_complex.((Pm' * Ω)')
Y_train, Y_test = Y[1:nbr_train, :], Y[(nbr_train + 1):nbr_states, :]

# Regression 
W = pinv(X_train) * Y_train
W_expected = pinv(S') * Pm
println("norm(W - W_expected): ", norm(W - W_expected))

Y_pred = X_test * W
mean_squared_error = mean((Y_test - Y_pred) .^ 2, dims = 1)

## ===================== Plotting ============================
plot_paulis = [
    (:σz, :σz)
]
plot_training_vs_test(plot_paulis, Y_test, Y_pred, Pm_dict, mean_squared_error, σE)
plot_test_and_prediction(plot_paulis, Y_test, Y_pred, Pm_dict, mean_squared_error)
plot_test(plot_paulis, Y_test, Pm_dict)
plot_training(plot_paulis, Y_train, Pm_dict)
test_S_row_space(S)
