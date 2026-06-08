function ridge_regression(X_train, Y_train, λ::Number = 0)
    if λ == 0
        return pinv(X_train) * Y_train
    end

    U, s, V = svd(X_train)
    D = Diagonal(s ./ (s .^ 2 .+ λ))
    W = V * D * U' * Y_train
    return W
end

function ridge_regression(X_train, Y_train, X_test, λ::Number = 0)
    W = ridge_regression(X_train, Y_train, λ)
    Y_pred = X_test * W
    return W, Y_pred
end

function degree_2_polynomial_feature_transformation(X)
    n_samples, n_features = size(X)
    hcat(X, X .^ 2,
        [X[:, i] .* X[:, j] for i in 1:n_features for j in (i + 1):n_features]...)
end

function split_train_test(X, Y, train_fraction = 0.5)
    nbr_train = round(Int, size(X, 1) * train_fraction)
    X_train, X_test = X[1:nbr_train, :], X[(nbr_train + 1):end, :]
    Y_train, Y_test = Y[1:nbr_train, :], Y[(nbr_train + 1):end, :]
    return X_train, X_test, Y_train, Y_test
end

mse(Y_true, Y_pred) = mean((Y_true - Y_pred) .^ 2)

function mse_prediction(S_SVD, Pm, σE, b)
    real.(diag(Pm' * S_SVD.V * diagm((b * σE^2) ./ (b .* S_SVD.S .^ 2 .+ σE^2)) * S_SVD.V' *
               Pm))
end