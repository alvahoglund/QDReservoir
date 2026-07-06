includet("..//CompareTasks/compare_tasks.jl")
includet("..//CompareTasks/plots_compare_tasks.jl")
includet("..//EntanglementWitness/entanglement_detection.jl")
includet("..//EntanglementWitness/plots_entanglement_detection.jl")
using JLD2, CairoMakie

## ============== Generate states ================

S = load("DefaultSystems/scrambling_map_A.jld2", "S")
sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")

nbr_sep_states = 10^4
nbr_ent_states = 10^4
nbr_train = (nbr_sep_states + nbr_ent_states) ÷ 2

Ω_sep_linear, Ω_ent_linear = linear_ew_states(sys, nbr_sep_states, nbr_ent_states)
Ω_sep_nonlinear, Ω_ent_nonlinear = nonlinear_ew_states(
    sys, nbr_sep_states, nbr_ent_states)
Ω_purity = random_mixed_states(nbr_sep_states + nbr_ent_states, sys)
Ω_spin = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main))
for i in 1:(nbr_sep_states + nbr_ent_states))

## ================ Compare performance of different tasks for varying noise levels ================
σE_list = 10 .^ range(-5, 0, length = 100)
linear_ew_results = vcat([linear_ew_performance(S, Ω_sep_linear, Ω_ent_linear, σE)
                          for σE in σE_list]...)
nonlinear_ew_results = vcat([nonlinear_ew_performance(
                                 S, Ω_sep_nonlinear, Ω_ent_nonlinear, σE)
                             for σE in σE_list]...)
purity_mse_list = vcat([purity_prediction_mse(S, Ω_purity, σE) for σE in σE_list]...)
spin_mse_list = vcat([spin_prediction_mse(S, Ω_spin, Pm, σE) for σE in σE_list]...)

##
svdvals_S = svdvals(S)
fig = Figure(size = (600, 250))
ax1 = Axis(
    fig[1, 1], xlabel = "Noise level (σE)", ylabel = "Fraction Incorrect/ MSE",
    title = "Performance of varying tasks", xscale = log10)
lines!(ax1, σE_list, linear_ew_results, label = "Linear Entanglement \nDetection")
lines!(ax1, σE_list, nonlinear_ew_results, label = "Nonlinear Entanglement \nDetection")
lines!(ax1, σE_list, purity_mse_list ./ maximum(purity_mse_list),
    label = "Purity Estimation")
lines!(ax1, σE_list, spin_mse_list ./ maximum(spin_mse_list),
    label = "Average Spin Estimation")
#vlines!(ax1, svdvals_S, linestyle = :dash, color = :grey,
#    label = "σS")
vlines!(ax1, [10^(-5), 10^(-2)], linestyle = :dash, color = :grey)

Legend(fig[1, 2], ax1)

save("Figures/compare_tasks_performance.png", fig)

## =============== LINEAR ENTANGLEMENT DETECTION =================
σE_samples = [10^(-5), 10^(-2)]
λ = 0
X_ent_linear = get_charge_measurements(S, Ω_ent_linear)
X_sep_linear = get_charge_measurements(S, Ω_sep_linear)

W_list = []
for σE in σE_samples
    X̃_ent = QDR.add_noise(X_ent_linear, σE)
    X̃_sep = QDR.add_noise(X_sep_linear, σE)
    W, _, _ = construct_EW(X̃_ent, X̃_sep, σE)
    push!(W_list, W)
end

fig_lew = Figure(size = (600, 450))

function add_W_heatmap_panel!(gl)
    rowsize!(fig_lew.layout, 1, Relative(0.3))  # top row: 30%
    ax11 = Axis(gl[1, 1])
    ax11.title = "Effective Weight Matrix, σE = $(format_σE(σE_samples[1]))"
    plot_heatmap_W_spin_basis!(ax11, W_list[1], S, sys)
    fig
    ax12 = Axis(gl[1, 2])
    ax12.title = "Effective Weight Matrix, σE = $(format_σE(σE_samples[2]))"
    hm12 = plot_heatmap_W_spin_basis!(ax12, W_list[2], S, sys)
    Colorbar(gl[1, 3], hm12, label = "Coefficient")
end
function add_linear_db_panel!(gl)
    Ω_sub_ent, Ω_sub_sep,
    W_sub_spin = project_on_sub_spin_basis(
        Ω_ent_linear, Ω_sep_linear, W_list[1])
    ax21 = plot_linear_db_spin_space!(gl, Ω_sub_sep, Ω_sub_ent, W_sub_spin,
        "Linear decision boundary, σE = $(format_σE(σE_samples[1]))")
    Legend(fig_lew[2, 2], ax21)
end
function add_werner_state_panel!(gl)
    ax22 = Axis(gl[1, 1])
    ax22.title = "Decision value for \n Singlet Werner state, σE = $(format_σE(σE_samples[1]))"
    test_werner_state!(ax22, [QDR.singlet], W_list[1], S, σE_samples[1])
end

fig1 = Figure(size = (800, 450))
add_W_heatmap_panel!(fig1[1, 1:2])
add_linear_db_panel!(fig1[2, 1])
add_werner_state_panel!(fig1[2, 2])
fig1
save("Figures/linear_entanglement_detection.png", fig_lew)

fig_W = Figure(size = (700, 300))
add_W_heatmap_panel!(fig_W[1, 1:2])
fig_W
save("Figures/linear_entanglement_detection_W.png", fig_W)
## =============== NONLINEAR ENTANGLEMENT DETECTION =================
X_ent_nonlinear = get_charge_measurements(S, Ω_ent_nonlinear)
X_sep_nonlinear = get_charge_measurements(S, Ω_sep_nonlinear)

σE = 10^-2
λ = 0

X̃_ent = QDR.add_noise(X_ent_nonlinear, σE)
X̃_sep = QDR.add_noise(X_sep_nonlinear, σE)

section_size = length(sys.grids.total) * 3
feature_transformation_alg = QDR.Polynomial2SectionFeatureTransformation(section_size)
W_noisy, Y_pred_noisy, Y_test = construct_EW(X̃_ent, X̃_sep, σE, feature_transformation_alg)
W_lownoise, Y_pred_lownoise,
_ = construct_EW(
    X_ent_nonlinear, X_sep_nonlinear, 1e-5, feature_transformation_alg)

Ω_sub_ent, Ω_sub_sep = project_on_sub_spin_basis(Ω_ent_nonlinear, Ω_sep_nonlinear)

fig_new = Figure(size = (800, 450))
gl1 = GridLayout(fig_new[1, 1])
plot_nonlinear_db_spin_space!(
    gl1, Ω_sub_sep, Ω_sub_ent, W_lownoise, feature_transformation_alg,
    title = "Nonlinear decision boundary, σE = $(format_σE(1e-5))")

gl2 = GridLayout(fig_new[1, 2])
ax = Axis(
    gl2[1, 1], title = "Decision value for \n Singlet Werner state, σE = $(format_σE(σE))")
fraction_correct = test_werner_state!(
    ax, [QDR.singlet, QDR.triplet_0, QDR.triplet_plus, QDR.triplet_minus], W_noisy, S, σE,
    feature_transformation_alg, state_labels = ["S", "T0", "T+", "T-"])
save("Figures/nonlinear_entanglement_detection.png", fig_new)

using GLMakie
fig_db = Figure(size = (800, 450))
gl1 = GridLayout(fig_db[1, 1])
plot_linear_db_spin_space!(gl1, Ω_sub_sep, Ω_sub_ent, W_sub_spin,
    "Linear decision boundary, σE = $(format_σE(σE_samples[1]))")
gl2 = GridLayout(fig_db[1, 2])
plot_nonlinear_db_spin_space!(
    gl2, Ω_sub_sep, Ω_sub_ent, W_lownoise, feature_transformation_alg,
    title = "Nonlinear decision boundary, σE = $(format_σE(1e-5))")
save("Figures/decision_boundaries.png", fig_db)