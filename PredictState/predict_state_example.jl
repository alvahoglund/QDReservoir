using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions
import QDReservoir as QDR

## ================= Functions ======================
function get_ham(grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.main)
    reservoir_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.res)
    interaction_parameters = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(grids, main_system_parameters,
        reservoir_parameters, interaction_parameters)
end

function trace_distance(ρ1, ρ2)
    return 0.5 * sum(svdvals(ρ1 - ρ2))
end

function trace_distances(Y_pred, Y_test)
    trace_distances = zeros(size(Y_pred, 1))
    for i in eachindex(size(Y_pred, 1))
        ρ_pred = reshape(Y_pred[i, :], 4, 4)
        ρ_test = reshape(Y_test[i, :], 4, 4)
        trace_distances[i] = trace_distance(ρ_pred, ρ_test)
    end
    return trace_distances
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
σE = 0
t = 100

measurements = map(m -> matrix_representation(m, sys.H_total),
    QDR.charge_probabilities(sys.grids.total))
Ω = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main)) for i in 1:nbr_states)
S = scrambling_map(sys, measurements, ground_state(hams.res),
    hams.total, t)

X = QDR.process_complex.((S * Ω)')
E = rand(Normal(0, σE), size(X))
X̃ = X + E

X̃_train, X̃_test = X̃[1:nbr_train, :], X̃[(nbr_train + 1):nbr_states, :]

Y = Ω'
Y_train, Y_test = Y[1:nbr_train, :], Y[(nbr_train + 1):nbr_states, :]

W = pinv(X̃_train) * Y_train
W_expected = pinv(S')
println("norm(W - W_expected): ", norm(W - W_expected))

Y_pred = X̃_test * W

trace_distances_list = trace_distances(Y_pred, Y_test)
mean_trace_distance = mean(trace_distances_list)