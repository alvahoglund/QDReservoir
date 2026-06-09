##
using QDReservoir
using LinearAlgebra, Statistics, Distributions, Random
import QDReservoir as QDR

function get_prod_states(nbr_sep_states, sys)
    stack(vec(density_matrix(QDR.random_product_state(sys.Hs_main, sys.H_main)))
    for i in 1:nbr_sep_states)
end

function get_sep_states(nbr_sep_states, sys)
    stack(vec(density_matrix(QDR.random_separable_state(
              rand(1:3), sys.Hs_main, sys.H_main)))
    for i in 1:nbr_sep_states)
end

function get_sep_states(nbr_sep_states, rank, sys)
    stack(vec(density_matrix(QDR.random_separable_state(
              rank, sys.Hs_main, sys.H_main)))
    for i in 1:nbr_sep_states)
end

function get_ent_states(nbr_ent_states, sys, state_names)
    p_list = range(0, 1 / 3, length = nbr_ent_states ÷ length(state_names))
    mapreduce(state -> stack(vec(QDR.werner_state(state, p, sys.H_main)) for p in p_list),
        hcat, state_names)
end

get_charge_measurements(S, Ω) = QDR.process_complex.((S * Ω)')

function construct_EW(X_ent, X_sep, σE::Number,
        feature_transformation_alg = QDR.IdentityFeatureTransformation())
    X̃_ent = QDR.add_noise(X_ent, σE)
    X̃_sep = QDR.add_noise(X_sep, σE)
    construct_EW(X̃_ent, X̃_sep, feature_transformation_alg)
end

function split_data(X̃_ent, X̃_sep)
    X̃ = vcat(X̃_sep, X̃_ent)
    Y = vcat(ones(size(X̃_sep, 1)), (-1) .* ones(size(X̃_ent, 1)))

    perm = randperm(size(X̃, 1))
    X̃ = X̃[perm, :]
    Y = Y[perm]

    nbr_train = size(X̃, 1) ÷ 2
    X̃_train, X̃_test = X̃[1:nbr_train, :], X̃[(nbr_train + 1):end, :]
    Y_train, Y_test = Y[1:nbr_train], Y[(nbr_train + 1):end]
    return X̃_train, X̃_test, Y_train, Y_test
end

function construct_EW(
        X̃_ent, X̃_sep, feature_transformation_alg = QDR.IdentityFeatureTransformation())
    X̃_train, X̃_test, Y_train, Y_test = split_data(
        X̃_ent, X̃_sep)
    X̃_train_poly = QDR.feature_transformation(X̃_train, feature_transformation_alg)
    X̃_test_poly = QDR.feature_transformation(X̃_test, feature_transformation_alg)
    W, Y_pred = QDR.ridge_regression(X̃_train_poly, Y_train, X̃_test_poly)
    return W, Y_pred, Y_test
end

function test_EW(
        X, σE, W, Y, feature_transformation_alg = QDR.IdentityFeatureTransformation())
    X_noisy = QDR.add_noise(X, σE)
    X_poly = QDR.feature_transformation(X_noisy, feature_transformation_alg)
    Y_pred = X_poly * W
    return get_fraction_correct(Y_pred, Y)
end

function get_ew_mse(
        X_ent, X_sep, σE, feature_transformation_alg = QDR.IdentityFeatureTransformation())
    W, Y_pred, Y_true = construct_EW(X_ent, X_sep, σE, feature_transformation_alg)
    return mse(Y_true, Y_pred)
end

function get_ew_fraction_correct(
        X_ent, X_sep, σE, feature_transformation_alg = QDR.IdentityFeatureTransformation())
    W, Y_pred, Y_true = construct_EW(X_ent, X_sep, σE, feature_transformation_alg)
    return get_fraction_correct(Y_true, Y_pred)
end

get_fraction_correct(Y_true, Y_pred) = mean((Y_true .> 0) .== (Y_pred .> 0))

function get_fraction_correct_classes(Y_true, Y_pred)
    sep_mask = Y_true .== 1
    ent_mask = Y_true .== -1
    Dict(
        "Separable" => mean(Y_pred[sep_mask] .> 0),
        "Entangled" => mean(Y_pred[ent_mask] .< 0)
    )
end

function get_linear_db(W_spin)
    x_range = range(-1, 1, length = 100)
    y_range = range(-1, 1, length = 100)
    b = W_spin[1]
    wx = W_spin[2]
    wy = W_spin[3]
    wz = W_spin[4]
    db_plane = [-(b + wx * x + wy * y) / wz for x in x_range, y in y_range]
    return x_range, y_range, db_plane
end

function get_sub_spin_basis(sys)
    Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
    Pm[:,
        [Pm_dict[(:σ0, :σ0)], Pm_dict[(:σx, :σx)],
            Pm_dict[(:σy, :σy)], Pm_dict[(:σz, :σz)]]]
end

function project_on_sub_spin_basis(Ω_ent, Ω_sep, W)
    Pm_sub = get_sub_spin_basis(sys)
    Ω_sub_ent = QDR.process_complex.(Ω_ent' * Pm_sub)
    Ω_sub_sep = QDR.process_complex.(Ω_sep' * Pm_sub)
    W_sub_spin = (QDR.process_complex.(S * Pm_sub))' * W .* (1 / 4)
    return Ω_sub_ent, Ω_sub_sep, W_sub_spin
end

function project_on_sub_spin_basis(Ω_ent, Ω_sep)
    Pm_sub = get_sub_spin_basis(sys)
    Ω_sub_ent = QDR.process_complex.(Ω_ent' * Pm_sub)
    Ω_sub_sep = QDR.process_complex.(Ω_sep' * Pm_sub)
    return Ω_sub_ent, Ω_sub_sep
end

function test_werner_state_against_noise(
        state_list, W_list, S, σE_list,
        feature_transformation_alg = QDR.IdentityFeatureTransformation())
    nbr_test_states = 1000
    Y = (-1) .* ones(nbr_test_states)
    X_states = [get_charge_measurements(S, get_ent_states(nbr_test_states, sys, [state]))
                for state in state_list]
    EW_performance = [[test_EW(X_state, σ_E, W, Y,
                           feature_transformation_alg)
                       for (σ_E, W) in zip(σE_list, W_list)] for X_state in X_states]
    return EW_performance
end

function test_separable_state_against_noise(
        W_list, S, σE_list, feature_transformation_alg = QDR.IdentityFeatureTransformation())
    nbr_test_states = 1000
    Y = ones(nbr_test_states)
    Ω_list = [get_sep_states(nbr_test_states, rank, sys) for rank in 1:4]
    X_sep_list = [get_charge_measurements(S, Ω) for Ω in Ω_list]

    EW_performance = [[test_EW(X_sep, σ_E, W, Y,
                           feature_transformation_alg)
                       for (σ_E, W) in zip(σE_list, W_list)] for X_sep in X_sep_list]
    return EW_performance
end