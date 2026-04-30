using QDReservoir, LinearAlgebra, Random, Statistics
import QDReservoir as QDR

# Set BLAS to single-threaded to avoid oversubscription
BLAS.set_num_threads(1)
## ===================== Functions ========================
clean_val(y) = map(x -> abs(x) < 1e-14 ? NaN
                        : x, y)

function get_ham(grids, ϵ_func_main, ϵ_func_res, ϵb_func,
        u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_parameters = QDR.set_dot_params(
        ϵ_func_main, ϵb_func, u_intra_func, grids.main)
    reservoir_parameters = QDR.set_dot_params(ϵ_func_res, ϵb_func, u_intra_func, grids.res)
    interaction_parameters = QDR.set_interaction_params(
        t_func, t_so_func, u_inter_func, grids.total)
    hamiltonians(grids, main_system_parameters,
        reservoir_parameters, interaction_parameters)
end

function get_ham(grids, parameters)
    get_ham(grids, parameters.ϵ_func_main, parameters.ϵ_func_res, parameters.ϵb_func,
        parameters.u_intra_func, parameters.t_func, parameters.t_so_func, parameters.u_inter_func)
end

function smallest_sv(sys, m_ops, hams_matrix, ψ_ground, t)
    S = scrambling_map(
        sys, m_ops, ψ_ground, hams_matrix.total, t)
    return minimum(svdvals(S))
end

## ===================== Functions for Singular values vs. reservoir electrons ========================

function avg_sv_vs_qn(nbr_dots_res, t, nbr_samples, parameters)
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = QDR.charge_probabilities(grid.total)
    n_qn = 2 * nbr_dots_res + 1

    sys_list = map(Base.Fix1(tight_binding_system, grid), 0:(n_qn - 1))
    m_ops_list = map(
        sys -> QDR.matrix_representation_ops(measurements, sys.H_total), sys_list)
    hams_symb = [get_ham(grid, parameters) for _ in 1:nbr_samples]

    hams_mat = Matrix{QDR.Hamiltonians}(undef, n_qn, nbr_samples)
    for idx in CartesianIndices(hams_mat)
        qn_res_idx, j = Tuple(idx)
        hams_mat[qn_res_idx, j] = QDR.matrix_representation_hams(
            hams_symb[j], sys_list[qn_res_idx])
    end
    ψ_res = map(ham_mat -> ground_state(ham_mat.res), hams_mat)
    sv_matrix = zeros(Float64, n_qn, nbr_samples)

    #Threads.@threads for idx in CartesianIndices(sv_matrix)
    for idx in CartesianIndices(sv_matrix)
        qn_res_idx, j = Tuple(idx)
        sv_matrix[idx] = smallest_sv(
            sys_list[qn_res_idx], m_ops_list[qn_res_idx],
            hams_mat[qn_res_idx, j], ψ_res[qn_res_idx, j], t)
    end

    mean_sv = vec(mean(sv_matrix, dims = 2))
    std_sv = vec(std(sv_matrix, dims = 2, corrected = false))
    median_sv = vec(median(sv_matrix, dims = 2))

    return mean_sv, std_sv, median_sv
end

function avg_sv_vs_res(nbr_dots_res_list, t, nbr_samples, parameters)
    avg_sv_dict = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
    for nbr_dots_res in nbr_dots_res_list
        println("Calculating for reservoir dots: $(nbr_dots_res)")
        avg_sv_list = avg_sv_vs_qn(nbr_dots_res, t, nbr_samples, parameters)
        avg_sv_dict[nbr_dots_res] = avg_sv_list
    end
    return avg_sv_dict
end

## ================= Smallest singular values vs parameter ========================

function avg_sv_vs_param(nbr_dots_res, qn_res, parameter_list, nbr_samples, t)
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = QDR.charge_probabilities(grid.total)
    time1 = time()
    sys = tight_binding_system(grid, qn_res)
    m_ops = QDR.matrix_representation_ops(measurements, sys.H_total)

    time2 = time()
    println("Set ops: $(time2 - time1)")

    hams_symb = map(
        Base.Fix1(get_ham, grid), [p for p in parameter_list, _ in 1:nbr_samples])
    hams_mat = map(Base.Fix2(QDR.matrix_representation_hams, sys), hams_symb)
    ψ_ground = map(ham_mat -> ground_state(ham_mat.res), hams_mat)

    time3 = time()
    println("Hamiltonians and ground state: $(time3- time2)")

    smallest_svs = zeros(Float64, length(parameter_list), nbr_samples)

    Threads.@threads :dynamic for idx in CartesianIndices(smallest_svs)
        #for idx in CartesianIndices(smallest_svs)
        i, j = Tuple(idx)
        smallest_svs[idx] = smallest_sv(sys, m_ops, hams_mat[i, j], ψ_ground[i, j], t)
    end

    time4 = time()
    println("Singular values: $(time4-time3)")
    mean_sv = vec(mean(smallest_svs, dims = 2))
    std_sv = vec(std(smallest_svs, dims = 2, corrected = false))
    median_sv = vec(median(smallest_svs, dims = 2))

    return mean_sv, std_sv, median_sv
end

function avg_sv_vs_param(reservoir_settings, parameter_list, nbr_samples, t)
    avg_sv_dict = Dict{
        Tuple{Int, Int}, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
    for (nbr_dots_res, qn_res) in reservoir_settings
        println("Calculating for reservoir dots: $(nbr_dots_res), reservoir electrons: $(qn_res)")
        avg_sv_list = avg_sv_vs_param(nbr_dots_res, qn_res, parameter_list, nbr_samples, t)
        avg_sv_dict[(nbr_dots_res, qn_res)] = avg_sv_list
    end
    return avg_sv_dict
end