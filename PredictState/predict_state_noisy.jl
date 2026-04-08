using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions
import QDReservoir as QDR
##

function trace_distance(ρ1, ρ2)
    return 0.5 * sum(svdvals(ρ1 - ρ2))
end

function trace_dist(Y_pred, Y_test)
    trace_distances_list = zeros(size(Y_pred, 1))
    for i in eachindex(size(Y_pred, 1))
        ρ_pred = reshape(Y_pred[i, :], 4, 4)
        ρ_test = reshape(Y_test[i, :], 4, 4)
        trace_distances_list[i] = trace_distance(ρ_pred, ρ_test)
    end
    return trace_distances_list
end

function add_noise(X, σE)
    E = rand(Normal(0, σE), size(X))
    return X + E
end

function predict_states(X_train, Y_train, X_test)
    W = pinv(X_train) * Y_train
    return X_test * W
end

function predict_states_with_noise(X_train, Y_train, X_test, σE)
    X_train_noisy = add_noise(X_train, σE)
    Y_pred = predict_states(X_train_noisy, Y_train, X_test)
    return Y_pred
end

function vary_noise(X_train, Y_train, X_test, Y_test, σE_list)
    trace_distances_list = zeros(length(σE_list))
    for (i, σE) in enumerate(σE_list)
        Y_pred = predict_states_with_noise(X_train, Y_train, X_test, σE)
        trace_distances_list[i] = mean(trace_dist(Y_pred, Y_test))
    end
    return trace_distances_list
end

function plot_trace_distance_vs_noise(σE_list, trace_distances_list)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "σE", ylabel = "Mean Trace Distance",
        xscale = log10)
    lines!(ax, σE_list, trace_distances_list)
    display(fig)
end

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

hams = QDR.matrix_representation_hams(
    get_ham(sys.grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func),
    sys)

nbr_states = 1000
nbr_train = nbr_states ÷ 2
nbr_test = nbr_states - nbr_train
t = 100

measurements = map(m -> matrix_representation(m, sys.H_total),
    QDR.charge_probabilities(sys.grids.total))
S = scrambling_map(sys, measurements, ground_state(hams.res),
    hams.total, t)
Ω = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main)) for i in 1:nbr_states)
X = QDR.process_complex.((S * Ω)')

X_train, X_test = X[1:nbr_train, :], X[(nbr_train + 1):nbr_states, :]

Y = Ω'
Y_train, Y_test = Y[1:nbr_train, :], Y[(nbr_train + 1):nbr_states, :]

##
σE_list = 10 .^ range(-8, -1, length = 10)
trace_distances_noise = vary_noise(X_train, Y_train, X_test, Y_test, σE_list)

plot_trace_distance_vs_noise(σE_list, trace_distances_noise)