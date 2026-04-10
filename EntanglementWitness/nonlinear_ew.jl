includet("ridge_classifier_EW.jl")

## ============= NONLINEAR ENTANGLEMENT WITNESS FOR WERNER STATES    =====================
sys, hams = default_system()
S = default_scrambling(sys, hams)

nbr_sep_states = 10^5
nbr_ent_states = 10^5
nbr_train = (nbr_sep_states + nbr_ent_states) ÷ 2

Ω_sep = get_sep_states(nbr_sep_states, sys)
states = [QDR.singlet, QDR.triplet_0, QDR.triplet_plus, QDR.triplet_minus]
Ω_ent = get_ent_states(nbr_ent_states, sys, states)
Ω = vcat(Ω_sep, Ω_ent)

X_ent = get_charge_measurements(S, Ω_ent)
X_sep = get_charge_measurements(S, Ω_sep)

## ============ Example of nonlinear Entanglement Witness for Werner states ==============
σE = 10^-5
λ = 0

X̃_ent = add_noise(σE, X_ent)
X̃_sep = add_noise(σE, X_sep)

feature_transformation_func = QDR.degree_2_polynomial_feature_transformation
W, Y_pred, Y_test = construct_EW(X_ent, X_sep, σE, feature_transformation_func)

plot_test_vs_pred_ew(Y_test, Y_pred)
Ω_sub_ent, Ω_sub_sep = project_on_sub_spin_basis(Ω_ent, Ω_sep)
plot_nonlinear_db_spin_space(
    Ω_sub_sep, Ω_sub_ent, W, feature_transformation_func)

Ω_ent_noisy = (X̃_ent * pinv(S'))'
Ω_sep_noisy = (X̃_sep * pinv(S'))'
Ω_sub_ent_noisy, Ω_sub_sep_noisy = project_on_sub_spin_basis(
    Ω_ent_noisy, Ω_sep_noisy)
plot_nonlinear_db_spin_space(
    Ω_sub_sep_noisy, Ω_sub_ent_noisy, W, feature_transformation_func)

test_werner_state(
    states, W, S,
    feature_transformation_func)

## =========== Plot accuracy of nonlinear EW for varying noise levels ================
σE_list = 10 .^ range(-7, 0, length = 20)

EW_list = [construct_EW(X_ent, X_sep, σE,
               QDR.degree_2_polynomial_feature_transformation)
           for σE in σE_list]

W_list = getindex.(EW_list, 1)
Y_pred_list = getindex.(EW_list, 2)
Y_test_list = getindex.(EW_list, 3)

# Plot accuracy of test data
fraction_correct_list = vcat([get_fraction_correct(Y_pred, Y_test)
                              for (Y_pred, Y_test) in zip(Y_pred_list, Y_test_list)]...)
plot_ew_fraction_correct(σE_list, fraction_correct_list)

# Plot accuracy of werner states against noise levels
EW_performance = test_werner_state_against_noise(
    states, W_list, S, σE_list, QDR.degree_2_polynomial_feature_transformation)

# Plot accuracy of separable states against noise levels
EW_performance_sep = test_separable_state_against_noise(
    W_list, S, σE_list, QDR.degree_2_polynomial_feature_transformation)

state_labels = ["Singlet", "Triplet 0", "Triplet +", "Triplet -",
    "Separable 1", "Separable 2", "Separable 3", "Separable 4"]
plot_state_performance_against_noise(
    state_labels, vcat(EW_performance, EW_performance_sep), σE_list)