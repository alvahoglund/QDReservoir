using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions
import QDReservoir as QDR
##
nbr_dots_res = 6
qn_res = 3
sys = tight_binding_system(2, nbr_dots_res, qn_res)
hams = QDR.matrix_representation_hams(hamiltonians(sys), sys)

nbr_states = 10^6
nbr_train = nbr_states ÷ 2
nbr_test = nbr_states - nbr_train
σE = 0

# ===================== Charge Measurements, X =============================
measurements = map(m -> matrix_representation(m, sys.H_total),
    QDR.single_charge_probabilities(sys.grid.total))
Ω = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main)) for i in 1:nbr_states)
S = scrambling_map(sys, measurements, ground_state(hams.hamiltonian_reservoir),
    hams.hamiltonian_total, [10, 100, 1000])
X = QDR.process_complex.((S * Ω)')
E = rand(Normal(0, σE), size(X))
X̃ = X + E

X_train, X_test = X̃[1:nbr_train, :], X̃[(nbr_train + 1):nbr_states, :]
# ===================== Spin Measurements, Y    =============================
Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
Y = QDR.process_complex.((Pm' * Ω)')
Y_train, Y_test = Y[1:nbr_train, :], Y[(nbr_train + 1):nbr_states, :]

# ===================== Regression ============================
W = pinv(X_train) * Y_train
W_expected = pinv(S') * Pm
println("norm(W - W_expected): ", norm(W - W_expected))

Y_pred = X_test * W
mean_squared_error = mean((Y_test - Y_pred) .^ 2, dims = 1)

# ===================== Plotting ============================
plot_paulis = [
    (:σ0, :σx),
    (:σ0, :σy)
]
x_vals = range(-1, 1, length = 100)
fig = Figure()
for (i, ps) in enumerate(plot_paulis)
    idx = Pm_dict[ps...]
    ax = Axis(fig[1, i],
        xlabel = "True value",
        ylabel = "Predicted value",
        title = "Measurement: $(ps[1]) ⊗ $(ps[2]), RMSE: $(round(sqrt(mean_squared_error[idx]), digits=4))")
    scatter!(ax, Y_test[1:1000:end, idx], Y_pred[1:1000:end, idx])
    lines!(ax, x_vals, x_vals, color = :black, label = "Optimal")
    axislegend(position = :rb)
end

display(fig)
