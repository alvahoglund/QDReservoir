using QDReservoir
using LinearAlgebra, Statistics, Distributions, Random
import QDReservoir as QDR
##
get_purity(Ω) = [real(dot(Ω[:, i], Ω[:, i])) for i in eachindex(Ω[1, :])]

function random_mixed_states(nbr_states, sys)
    p_list = rand(nbr_states)
    mapreduce(
        i -> vec((1 - p_list[i]) * density_matrix(QDR.random_state(sys.H_main)) +
                 p_list[i] * QDR.max_mixed_state(sys.H_main)),
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
    feature_transformation_func = QDR.degree_2_polynomial_feature_transformation
    X̃_train_poly = feature_transformation_func(X̃_train)
    X̃_test_poly = feature_transformation_func(X̃_test)
    W, Y_pred = QDR.ridge_regression(X̃_train_poly, Y_train, X̃_test_poly, λ)
    return W, Y_pred, Y_test
end

function get_purity_mse(X, Y, σE)
    W, Y_pred, Y_true = estimate_purity(X, Y, σE)
    return QDR.mse(Y_true, Y_pred)
end