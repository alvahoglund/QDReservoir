using QDReservoir
using LinearAlgebra, Statistics, Distributions
import QDReservoir as QDR

function randomize_hs_states(sys, nbr_states)
    stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main)) for _ in 1:nbr_states)
end

function train_model(X, Y, σE)
    X̃ = QDR.add_noise(X, σE)
    X_train, X_test, Y_train, Y_test = QDR.split_train_test(X̃, Y)
    W, Y_pred = QDR.ridge_regression(X_train, Y_train, X_test)
    return (; W, X_test, Y_test, Y_pred)
end

function train_model(Ω, S, Pm, σE)
    X = QDR.process_complex.((S * Ω)')
    Y = QDR.process_complex.((Pm' * Ω)')
    return train_model(X, Y, σE)
end

function get_mse(X, Y, σE)
    result = train_model(X, Y, σE)
    return vec(mean((result.Y_test - result.Y_pred) .^ 2, dims = 1))
end

function SV_overlap(S_SVD, Pm)
    abs2.(S_SVD.V' * Pm)
end

function get_A(S, σE, b)
    S_SVD = svd(S)
    d = b .* S_SVD.S .^ 2 ./ (b .* S_SVD.S .^ 2 .+ σE^2)
    S_SVD.U * Diagonal(d) * S_SVD.U'
end

function test_weights(W, Pm, S, σE, b)
    A = get_A(S, σE, b)
    R = A * pinv(S') * Pm
    return norm(W - R, 2) / norm(R, 2)
end

function evaluate_model(Ω_train, X_test, Y_test, S, Pm, σE, b)
    X_train = QDR.process_complex.((S * Ω_train)')
    Y_train = QDR.process_complex.((Pm' * Ω_train)')
    X̃_train = QDR.add_noise(X_train, σE)
    W = QDR.ridge_regression(X̃_train, Y_train)
    mse = vec(mean((Y_test - X_test * W) .^ 2, dims = 1))
    weight_err = test_weights(W, Pm, S, σE, b)
    return (; mse, weight_err)
end

function vary_training_data(sys, X_test, Y_test, S, Pm, σE, b, nbr_states_list)
    map(nbr_states_list) do nbr_states_train
        Ω_train = randomize_hs_states(sys, nbr_states_train)
        evaluate_model(Ω_train, X_test, Y_test, S, Pm, σE, b)
    end
end
