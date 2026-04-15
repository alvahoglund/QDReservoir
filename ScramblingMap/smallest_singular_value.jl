using QDReservoir, LinearAlgebra, Random, Statistics
import QDReservoir as QDR

# Set BLAS to single-threaded to avoid oversubscription
#BLAS.set_num_threads(1)
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

function smallest_sv(sys, m_ops, ham_symb, t)
    hams_matrix = QDR.matrix_representation_hams(ham_symb, sys)
    ψ_res = ground_state(hams_matrix.res)
    S = scrambling_map(
        sys, m_ops, ψ_res, hams_matrix.total, t)
    return minimum(svdvals(S))
end

## ===================== Functions for Singular values vs. reservoir electrons ========================

function avg_sv_vs_qn(nbr_dots_res, t, nbr_samples, parameters)
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = QDR.charge_probabilities(grid.total)
    n_qn = 2 * nbr_dots_res + 1

    sys_list = [tight_binding_system(grid, qn_res) for qn_res in 0:(n_qn - 1)]
    m_ops_list = [QDR.matrix_representation_ops(measurements, sys.H_total)
                  for sys in sys_list]

    randomized_hams = [get_ham(grid, parameters) for _ in 1:nbr_samples]

    sv_matrix = zeros(Float64, n_qn, nbr_samples)

    #Threads.@threads for idx in CartesianIndices(sv_matrix)
    for idx in CartesianIndices(sv_matrix)
        qn_res_idx, j = Tuple(idx)
        sv_matrix[idx] = smallest_sv(
            sys_list[qn_res_idx], m_ops_list[qn_res_idx],
            randomized_hams[j], t)
    end

    mean_sv = vec(mean(sv_matrix, dims = 2))
    std_sv = vec(std(sv_matrix, dims = 2, corrected = false))

    return mean_sv, std_sv
end

function avg_sv_vs_res(nbr_dots_res_list, t, nbr_samples, parameters)
    avg_sv_dict = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
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

    sys = tight_binding_system(grid, qn_res)
    m_ops = QDR.matrix_representation_ops(measurements, sys.H_total)

    hams_symb = [get_ham(grid, parameter_list[i])
                 for i in eachindex(parameter_list), _ in 1:nbr_samples]
    #hams_symb = [hamiltonians(grid) for _ in eachindex(parameter_list), _ in 1:nbr_samples]

    smallest_svs = Matrix{Float64}(undef, length(parameter_list), nbr_samples)

    #Threads.@threads for idx in CartesianIndices(smallest_svs)
    for idx in CartesianIndices(smallest_svs)
        i, j = Tuple(idx)
        smallest_svs[idx] = smallest_sv(sys, m_ops, hams_symb[i, j], t)
    end

    mean_sv = [sum(smallest_svs[i, :]) / nbr_samples for i in 1:length(parameter_list)]
    std_sv = [sqrt(sum(smallest_svs[i, j]^2 for j in 1:nbr_samples) / nbr_samples -
                   mean_sv[i]^2)
              for i in 1:length(parameter_list)]

    return mean_sv, std_sv
end

function avg_sv_vs_param(reservoir_settings, parameter_list, nbr_samples, t)
    avg_sv_dict = Dict{Tuple{Int, Int}, Tuple{Vector{Float64}, Vector{Float64}}}()
    for (nbr_dots_res, qn_res) in reservoir_settings
        println("Calculating for reservoir dots: $(nbr_dots_res), reservoir electrons: $(qn_res)")
        avg_sv_list = avg_sv_vs_param(nbr_dots_res, qn_res, parameter_list, nbr_samples, t)
        avg_sv_dict[(nbr_dots_res, qn_res)] = avg_sv_list
    end
    return avg_sv_dict
end