using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions
import QDReservoir as QDR
##
function split_train_test(X, Y)
    nbr_train = size(X)[1] ÷ 2
    X_train, X_test = X[1:nbr_train, :], X[nbr_train+1:end, :]
    Y_train, Y_test = Y[1:nbr_train, :], Y[nbr_train+1:end, :]
    return X_train, X_test, Y_train, Y_test
end

function get_mse(Ω, S, Pm, σE)
    X = QDR.process_complex.((S * Ω)')
    E = rand(Normal(0, σE), size(X))
    X̃ = X + E

    Y = QDR.process_complex.((Pm' * Ω)')

    X_train, X_test, Y_train, Y_test = split_train_test(X̃, Y)

    W = pinv(X_train) * Y_train

    Y_pred = X_test * W
    return mean((Y_test - Y_pred) .^ 2, dims=1)
end

mse_prediction(S_SVD, Pm, σE, b) = real.(diag(Pm' * S_SVD.V * diagm((b * σE^2) ./ (b .* S_SVD.S .^ 2 .+ σE^2)) * S_SVD.V' * Pm))

SV_overlap(S_SVD, Pm) = abs2.(S_SVD.V' * Pm)

function plot_varying_noise(Ω, S, Pm, Pm_dict, ps_list)
    S_SVD = svd(S)

    σE_list = 10 .^ range(-7, 0, length=30)
    mse_list = vcat([get_mse(Ω, S, Pm, σE) for σE in σE_list]...)

    b = 0.0147
    sv_overlaps = SV_overlap(S_SVD, Pm)
    mse_pred_list = vcat([transpose(mse_prediction(S_SVD, Pm, σE, b)) for σE in σE_list]...)

    fig = Figure()
    for (i, ps) in enumerate(ps_list)
        idx = Pm_dict[ps...]
        ax = Axis(fig[i, 1], xlabel="Noise level (σE)", ylabel="Mean Squared Error", title="Measurement: $(ps[1]) ⊗ $(ps[2])",
            xscale=log10)
        lines!(ax, σE_list, mse_list[:, idx], label="MSE")
        lines!(ax, σE_list, mse_pred_list[:, idx], label="Predicted MSE")
        vlines!(ax, sqrt(b) .* S_SVD.S, color=sv_overlaps[:, idx], colormap=:Blues, colorrange=(-0.1, maximum(sv_overlaps[:, idx])), linestyle=:dash, label="√b*σS")
        axislegend(position=:lt)
    end
    display(fig)
end

# ====== Choose system parameters ======
nbr_dots_res = 3
qn_res = 1
sys = tight_binding_system(2, nbr_dots_res, qn_res)
hams = QDR.matrix_representation_hams(QDR.hamiltonians(sys), sys)
ρ_res = ground_state(hams.hamiltonian_reservoir)
t_list = [10, 20]

nbr_states = 10^4
σE = 0
measurements = QDR.charge_measurements(sys)

Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
Ω = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main)) for i in 1:nbr_states)
S = scrambling_map(sys, measurements, ρ_res, hams.hamiltonian_total, t_list)

# ===== PLOT VARYING NOISE =========== 
ps_list = [
    (:σx, :σx)
    (:σ0, :σy)
]
plot_varying_noise(Ω, S, Pm, Pm_dict, ps_list)
