using QDReservoir
using LinearAlgebra, Statistics, Distributions, Random
import QDReservoir as QDR
##
get_purity(Ω) = [real(dot(Ω[:, i], Ω[:, i])) for i in eachindex(Ω[1, :])]

function random_mixed_states(nbr_states, H_main)
    p_list = rand(nbr_states)
    mapreduce(
        i -> vec((1 - p_list[i]) * density_matrix(QDR.random_state(H_main)) +
                 p_list[i] * QDR.max_mixed_state(H_main)),
        hcat, 1:nbr_states)
end

function get_purity_mse(X, Y, σE)
    W, Y_pred, Y_true = predict_purity(X, Y, σE)
    return QDR.mse(Y_true, Y_pred)
end

get_purity(Ω) = [real(dot(Ω[:, i], Ω[:, i])) for i in eachindex(Ω[1, :])]

function estimate_purity(X, Y, σE)
    λ = 0
    X̃ = QDR.add_noise(X, σE)
    X̃_train, X̃_test, Y_train, Y_test = QDR.split_train_test(X̃, Y)
    feature_transformation_alg = QDR.Polynomial2SectionFeatureTransformation(24)
    X̃_train_poly = QDR.feature_transformation(X̃_train, feature_transformation_alg)
    X̃_test_poly = QDR.feature_transformation(X̃_test, feature_transformation_alg)
    W, Y_pred = QDR.ridge_regression(X̃_train_poly, Y_train, X̃_test_poly, λ)
    return W, Y_pred, Y_test
end

function purity_from_state_estimation(X, Ω, σE)
    Y = Ω'
    X̃ = QDR.add_noise(X, σE)
    X̃_train, X̃_test, Y_train, Y_test = QDR.split_train_test(X̃, Y)
    W, Y_pred = QDR.ridge_regression(X̃_train, Y_train, X̃_test, 0)
    purity_test = get_purity(Y_test')
    purity_pred = get_purity(Y_pred')
    return purity_pred, purity_test
end

function purity_from_state_estimation_mse(X, Ω, σE)
    purity_pred, purity_test = purity_from_state_estimation(X, Ω, σE)
    return QDR.mse(purity_test, purity_pred)
end

function get_purity_mse(X, Y, σE)
    W, Y_pred, Y_true = estimate_purity(X, Y, σE)
    return QDR.mse(Y_true, Y_pred)
end

function rand_S()
    nbr_dots_res = rand(2:6)
    qn_res = rand(0:(2 * nbr_dots_res))
    sys = tight_binding_system(2, nbr_dots_res, qn_res)
    seed = 1323
    hams = QDR.matrix_representation_hams(QDR.hamiltonians(sys.grids, seed), sys)
    ρ_res = ground_state(hams.res)
    t_list = [100, 200]
    measurements = QDR.charge_probabilities(sys)
    S = scrambling_map(sys, measurements, ρ_res, hams.total, t_list)
    return S, sys
end

function get_purity_mse_stats(X_list, Y, σE_list)
    mse_mat = Matrix{Float64}(undef, length(σE_list), length(X_list))
    Threads.@threads for I in CartesianIndices(mse_mat)
        i, j = Tuple(I)
        mse_mat[I] = get_purity_mse(X_list[j], Y, σE_list[i])
    end
    return mse_mat
end

function get_purity_from_state_mse_stats(X_list, Ω, σE_list)
    mse_mat_s = Matrix{Float64}(undef, length(σE_list), length(X_list))
    Threads.@threads for I in CartesianIndices(mse_mat_s)
        i, j = Tuple(I)
        mse_mat_s[I] = purity_from_state_estimation_mse(X_list[j], Ω, σE_list[i])
    end
    return mse_mat_s
end