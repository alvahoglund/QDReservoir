function ridge_regression(X_train, Y_train, X_test, λ = 0)
    U, s, V = svd(X_train)
    d = s ./ (s .^ 2 .+ λ)
    W = V * Diagonal(d) * (U' * Y_train)
    Y_pred = X_test * W
    return W, Y_pred
end

function degree_2_polynomial_feature_transformation(X)
    n_samples, n_features = size(X)
    hcat(X, X .^ 2,
        [X[:, i] .* X[:, j] for i in 1:n_features for j in (i + 1):n_features]...)
end