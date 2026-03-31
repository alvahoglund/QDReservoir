##
using QDReservoir
using LinearAlgebra, Statistics, GLMakie, Distributions, Random
import QDReservoir as QDR

## ======================= Plotting Functions =============================
function heatmap_W_spin_basis(W, sys)
    Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)
    W_spin_basis = (QDR.process_complex.(S * Pm))' * W .* (1 / 4)
    W_spin_basis

    fig, ax, hm = heatmap(
        1:4, 1:4, reshape(W_spin_basis, (4, 4)), colormap = :bam, colorrange = [-1, 1])

    ax.xticks = (1:4, ["σ0", "σx", "σy", "σz"])
    ax.yticks = (1:4, ["σ0", "σx", " σy", "σz"])
    Colorbar(fig[:, end + 1], hm, label = "Coefficient")
    display(fig)
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

function project_on_3D_spin_space(Ω_ent, Ω_sep, W)
    Pm_sub = get_sub_spin_basis(sys)
    Ω_sub_ent = QDR.process_complex.(Ω_ent' * Pm_sub)
    Ω_sub_sep = QDR.process_complex.(Ω_sep' * Pm_sub)
    W_sub_spin = (QDR.process_complex.(S * Pm_sub))' * W .* (1 / 4)
    return Ω_sub_ent, Ω_sub_sep, W_sub_spin
end

function plot_3D_spin_space(Ω_sep_spin, Ω_ent_spin, W_spin)
    fig = Figure()
    ax = Axis3(fig[1, 1], xlabel = "XX", ylabel = "YY", zlabel = "ZZ",
        title = "Ridge regression classification in XX, YY, ZZ space")
    plot_states_sep = 10
    plot_states_ent = 1000
    scatter!(ax,
        Ω_sep_spin[1:plot_states_sep:end, 2],
        Ω_sep_spin[1:plot_states_sep:end, 3],
        Ω_sep_spin[1:plot_states_sep:end, 4],
        label = "Separable states", markersize = 5
    )

    scatter!(ax,
        Ω_ent_spin[1:plot_states_ent:end, 2],
        Ω_ent_spin[1:plot_states_ent:end, 3],
        Ω_ent_spin[1:plot_states_ent:end, 4],
        label = "Entangled states", markersize = 5
    )
    x_range, y_range, db_plane = get_linear_db(W_spin)
    surface!(ax, x_range, y_range, db_plane, label = "Decision boundary")

    display(fig)
end
function plot_test_vs_pred(Y_test, Y_pred)
    #Sort first on Y_test, then on Y_pred
    sort_indices = sortperm(Y_pred)
    Y_test = Y_test[sort_indices]
    Y_pred = Y_pred[sort_indices]
    x_range = range(-1, 1, length = length(Y_pred))
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Sorted index", ylabel = "Label value",
        title = "Ridge regression for entanglement witness, fraction correct = $(round(get_fraction_correct(Y_test, Y_pred), digits = 4))")
    scatter!(ax, x_range, tanh.(Y_pred * 3), label = "Predicted labels (tanh scaled)")
    scatter!(ax, x_range, Y_test, label = "True labels")
    scatter!(ax, x_range, Y_pred, label = "Predicted labels")
    axislegend(position = :lt)
    display(fig)
end
get_fraction_correct(Y_true, Y_pred) = mean((Y_true .> 0) .== (Y_pred .> 0))

function get_ham(grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.main)
    reservoir_parameters = QDR.set_dot_params(ϵ_func, ϵb_func, u_intra_func, grids.res)
    interaction_parameters = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(grids, main_system_parameters,
        reservoir_parameters, interaction_parameters)
end

## ================ System parameters
ϵ_func() = 0.5
ϵb_func() = [0, 0, 1]
u_intra_func() = rand() + 10
t_func() = rand()
t_so_func() = 0.1 * rand()
u_inter_func() = rand()

nbr_dots_res = 6
qn_res = 3
sys = tight_binding_system(2, nbr_dots_res, qn_res)

seed = 1234
Random.seed!(seed)
hams = QDR.matrix_representation_hams(
    get_ham(sys.grids, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func),
    sys)
σE = 10^-3
t = 100

measurements = map(m -> matrix_representation(m, sys.H_total),
    QDR.charge_probabilities(sys.grids.total))
S = scrambling_map(sys, measurements, ground_state(hams.res),
    hams.total, t)

# Generate data
nbr_sep_states = 10^5
nbr_ent_states = 10^5
nbr_train = (nbr_sep_states + nbr_ent_states) ÷ 2

Ω_sep = stack(vec(density_matrix(QDR.random_product_state(sys.Hs_main, sys.H_main)))
for i in 1:nbr_sep_states)
p_list = range(0, 1 / 3, length = nbr_ent_states)
Ω_ent = stack(vec(QDR.werner_state(QDR.singlet, p, sys.H_main)) for p in p_list)
Ω = hcat(Ω_sep, Ω_ent)

X_ent = QDR.process_complex.((S * Ω_ent)')
X̃_ent = X_ent + rand(Normal(0, σE), size(X_ent))
X_sep = QDR.process_complex.((S * Ω_sep)')
X̃_sep = X_sep + rand(Normal(0, σE), size(X_sep))
X = vcat(X_sep, X_ent)
X̃ = vcat(X̃_sep, X̃_ent)
# Shuffle data in X
perm = randperm(size(X, 1))
X = X[perm, :]
X̃ = X̃[perm, :]
X_train, X_test = X[1:nbr_train, :], X[(nbr_train + 1):end, :]
X̃_train, X̃_test = X̃[1:nbr_train, :], X̃[(nbr_train + 1):end, :]

Y = vcat((-1) .* ones(size(Ω_sep)[2]), ones(size(Ω_ent)[2]))[perm]
Y_train, Y_test = Y[1:nbr_train], Y[(nbr_train + 1):end]
λ = 0
W = (X̃_train' * X̃_train + λ * I) \ (X̃_train' * Y_train)
Y_pred = X̃_test * W

## Plots
heatmap_W_spin_basis(W, sys)
plot_test_vs_pred(Y_test, Y_pred)

Ω_sub_ent, Ω_sub_sep, W_sub_spin = project_on_3D_spin_space(Ω_ent, Ω_sep, W)
plot_3D_spin_space(Ω_sub_sep, Ω_sub_ent, W_sub_spin)

Ω_ent_noisy = (X̃_ent * pinv(S'))'
Ω_sep_noisy = (X̃_sep * pinv(S'))'
Ω_sub_ent_noisy, Ω_sub_sep_noisy, W_sub_spin = project_on_3D_spin_space(
    Ω_ent_noisy, Ω_sep_noisy, W)
plot_3D_spin_space(Ω_sub_sep_noisy, Ω_sub_ent_noisy, W_sub_spin)