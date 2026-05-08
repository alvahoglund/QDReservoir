using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions
import QDReservoir as QDR
##
function split_train_test(X, Y)
    nbr_train = size(X)[1] ÷ 2
    X_train, X_test = X[1:nbr_train, :], X[(nbr_train + 1):end, :]
    Y_train, Y_test = Y[1:nbr_train, :], Y[(nbr_train + 1):end, :]
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
    return mean((Y_test - Y_pred) .^ 2, dims = 1)
end

function mse_prediction(S_SVD, Pm, σE, b)
    real.(diag(Pm' * S_SVD.V * diagm((b * σE^2) ./ (b .* S_SVD.S .^ 2 .+ σE^2)) * S_SVD.V' *
               Pm))
end

SV_overlap(S_SVD, Pm) = abs2.(S_SVD.V' * Pm)

function plot_varying_noise(Ω, S, Pm, Pm_dict, ps_list)
    S_SVD = svd(S)

    σE_list = 10 .^ range(-7, 0, length = 30)
    mse_list = vcat([get_mse(Ω, S, Pm, σE) for σE in σE_list]...)

    b = 0.0147
    sv_overlaps = SV_overlap(S_SVD, Pm)
    mse_pred_list = vcat([transpose(mse_prediction(S_SVD, Pm, σE, b)) for σE in σE_list]...)

    fig = Figure(size = (700, 400))
    for (i, ps) in enumerate(ps_list)
        idx = Pm_dict[ps...]
        ax = Axis(fig[i, 1], xlabel = "Noise level (σE)", ylabel = "Mean Squared Error",
            title = "Measurement: $(ps[1]) ⊗ $(ps[2])",
            xscale = log10)
        lines!(ax, σE_list, mse_list[:, idx], label = "MSE")
        #vlines!(ax, [10^-4, 10^-2], linestyle = :dash, color = :grey)
        lines!(ax, σE_list, mse_pred_list[:, idx], label = "Predicted MSE")
        vlines!(ax, sqrt(b) .* S_SVD.S, color = sv_overlaps[:, idx],
            colormap = :Blues, colorrange = (-0.5, maximum(sv_overlaps[:, idx])),
            linestyle = :dash, label = "√b*σS")
        axislegend(position = :lt)
    end
    display(fig)
end
#
# ====== Choose system parameters ======
#br_dots_res = 6
#n_res = 3
#ys = tight_binding_system(2, nbr_dots_res, qn_res)
#eed = 1234
#ams = QDR.matrix_representation_hams(QDR.hamiltonians(sys.grids, seed), sys)
#_res = ground_state(hams.res)
#_list = [100]
#
#br_states = 10000
#easurements = QDR.charge_measurements(sys)
#
#m, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
# = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main)) for i in 1:nbr_states)
# = scrambling_map(sys, measurements, ρ_res, hams.total, t_list)
#
# ===== PLOT VARYING NOISE =========== 
#s_list = [(:σz, :σz)]
#lot_varying_noise(Ω, S, Pm, Pm_dict, ps_list)
#