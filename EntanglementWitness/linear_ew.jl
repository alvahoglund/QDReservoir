includet("ridge_classifier_EW.jl")

## 
sys, hams = default_system()
S = default_scrambling(sys, hams)

nbr_sep_states = 10^5
nbr_ent_states = 10^5
nbr_train = (nbr_sep_states + nbr_ent_states) ÷ 2

Ω_sep = get_prod_states(nbr_sep_states, sys)
state_names = [QDR.singlet]
Ω_ent = get_ent_states(nbr_ent_states, sys, state_names)
Ω = vcat(Ω_sep, Ω_ent)

X_ent = get_charge_measurements(S, Ω_ent)
X_sep = get_charge_measurements(S, Ω_sep)

## ============  Example of linear Entanglement Witness =====================

σE = 10^-2
λ = 0
X̃_ent = add_noise(σE, X_ent)
X̃_sep = add_noise(σE, X_sep)
W, Y_pred, Y_test = construct_EW(X_ent, X_sep, σE)

plot_bars_W_spin_basis([W], S, sys, [σE])
plot_heatmap_W_spin_basis(W, S, sys)
plot_test_vs_pred_ew(Y_test, Y_pred)

Ω_sub_ent, Ω_sub_sep, W_sub_spin = project_on_sub_spin_basis(Ω_ent, Ω_sep, W)
plot_linear_db_spin_space(Ω_sub_sep, Ω_sub_ent, W_sub_spin)

Ω_ent_noisy = (X̃_ent * pinv(S'))'
Ω_sep_noisy = (X̃_sep * pinv(S'))'
Ω_sub_ent_noisy, Ω_sub_sep_noisy, W_sub_spin = project_on_sub_spin_basis(
    Ω_ent_noisy, Ω_sep_noisy, W)
plot_linear_db_spin_space(Ω_sub_sep_noisy, Ω_sub_ent_noisy, W_sub_spin)
test_werner_state(state_names, W, S)

##Plot accuracy of EW for varying noise levels

σE_list = 10 .^ range(-10, 0, length = 50)
fraction_correct_list = vcat([get_ew_fraction_correct(X_ent, X_sep, σE) for σE in σE_list]...)
plot_ew_fraction_correct(σE_list, fraction_correct_list, [10^-10, 10^-4, 10^-2])

## Plot EW for varying noise levels

σE_list = [0, 10^-4, 10^-2]
W_list = [construct_EW(X_ent, X_sep, σE)[1] for σE in σE_list]
plot_bars_W_spin_basis(W_list, S, sys, σE_list)
