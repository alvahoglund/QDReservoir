
includet("../EntanglementWitness/entanglement_detection.jl")
includet("../Purity/estimate_purity.jl")
includet("../PredictSpin/estimate_spin.jl")
##
BLAS.set_num_threads(1)
function linear_ew_states(sys, nbr_sep_states, nbr_ent_states)
    Ω_sep = get_prod_states(nbr_sep_states, sys)
    state_names = [QDR.singlet]
    Ω_ent = get_ent_states(nbr_ent_states, sys, state_names)
    return Ω_sep, Ω_ent
end

function nonlinear_ew_states(sys, nbr_sep_states, nbr_ent_states)
    Ω_sep = get_sep_states(nbr_sep_states, sys)
    states = [QDR.singlet, QDR.triplet_0, QDR.triplet_plus, QDR.triplet_minus]
    Ω_ent = get_ent_states(nbr_ent_states, sys, states)
    return Ω_sep, Ω_ent
end

function linear_ew_performance(S, Ω_sep, Ω_ent, σE)
    X_sep = get_charge_measurements(S, Ω_sep)
    X_ent = get_charge_measurements(S, Ω_ent)
    return 1 - get_ew_fraction_correct(X_ent, X_sep, σE)
end

function nonlinear_ew_performance(S, Ω_sep, Ω_ent, σE)
    X_sep = get_charge_measurements(S, Ω_sep)
    X_ent = get_charge_measurements(S, Ω_ent)
    return 1 - get_ew_fraction_correct(
        X_ent, X_sep, σE, QDR.Polynomial2SectionFeatureTransformation(24))
end
function purity_prediction_mse(S, Ω, σE)
    X = get_charge_measurements(S, Ω)
    Y = get_purity(Ω)
    return get_purity_mse(X, Y, σE)
end
function spin_prediction_mse(S, Ω, Pm, σE)
    X = get_charge_measurements(S, Ω)
    Y = QDR.process_complex.((Pm' * Ω)')
    return mean(get_mse(X, Y, σE))
end
function set_ham(
        grids, ϵ_func_main, ϵ_func_res, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_params = QDR.set_dot_params(
        ϵ_func_main, ϵb_func, u_intra_func, grids.main)
    res_params = QDR.set_dot_params(
        ϵ_func_res, ϵb_func, u_intra_func, grids.res)
    interaction_params = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(
        grids, main_system_params, res_params, interaction_params)
end

function randomize_system()
    nbr_dots_res = rand(4:6)
    qn_res = rand(0:nbr_dots_res)
    sys = QDR.tight_binding_system(2, nbr_dots_res, qn_res)
    hams = hamiltonians(sys.grids)
    nbr_t = rand(2:4)
    t = [100 * rand() for _ in 1:nbr_t]
    return sys, hams, t
end

function get_scrambling_map(sys, hams, t)
    hams = QDR.matrix_representation_hams(hams, sys)
    ψ_ground = ground_state(hams.res)
    m_ops = QDR.matrix_representation_ops(
        QDR.charge_probabilities(sys.grids.total), sys.H_total)
    return QDR.scrambling_map(
        sys, m_ops, ψ_ground, hams.total, t, QDR.PureStateSteppingPropagatorAlg())
end

function get_performances(S, Ω_sep_linear, Ω_ent_linear, Ω_sep_nonlinear,
        Ω_ent_nonlinear, Ω_purity, Ω_spin, Pm, σE)
    linear_ew_results = linear_ew_performance(S, Ω_sep_linear, Ω_ent_linear, σE)
    nonlinear_ew_results = nonlinear_ew_performance(S, Ω_sep_nonlinear, Ω_ent_nonlinear, σE)
    purity_mse = purity_prediction_mse(S, Ω_purity, σE)
    spin_mse = spin_prediction_mse(S, Ω_spin, Pm, σE)
    return linear_ew_results, nonlinear_ew_results, purity_mse, spin_mse
end

function get_performances_matrix(S_list, Ω_sep_linear, Ω_ent_linear, Ω_sep_nonlinear,
        Ω_ent_nonlinear, Ω_purity, Ω_spin, Pm, σE_list)
    n_S = length(S_list)
    n_σE = length(σE_list)
    linear_ew_results = Matrix{Float64}(undef, n_S, n_σE)
    nonlinear_ew_results = Matrix{Float64}(undef, n_S, n_σE)
    purity_mse_results = Matrix{Float64}(undef, n_S, n_σE)
    spin_mse_results = Matrix{Float64}(undef, n_S, n_σE)
    Threads.@threads :dynamic for idx in CartesianIndices(linear_ew_results)
        i, j = Tuple(idx)
        S = S_list[i]
        σE = σE_list[j]
        linear_ew_results[i, j], nonlinear_ew_results[i, j], purity_mse_results[i, j], spin_mse_results[i, j] = get_performances(
            S, Ω_sep_linear, Ω_ent_linear, Ω_sep_nonlinear,
            Ω_ent_nonlinear, Ω_purity, Ω_spin, Pm, σE)
    end
    return linear_ew_results, nonlinear_ew_results, purity_mse_results, spin_mse_results
end
