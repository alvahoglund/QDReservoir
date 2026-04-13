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

function smallest_sv(grid, qn_res, hams, t, measurements)
    # For a given hamiltonian an qn_number, 
    # compute the smallest singular value of the scrambling map
    sys = tight_binding_system(grid, qn_res)
    m_ops = QDR.matrix_representation_ops(measurements, sys.H_total)
    hams = QDR.matrix_representation_hams(hams, sys)
    S = scrambling_map(sys, m_ops, ground_state(hams.res), hams.total, t)
    return minimum(svdvals(S))
end

## ===================== Functions for Singular values vs. reservoir electrons ========================

function avg_sv_vs_qn(nbr_dots_res, t, nbr_samples, parameters)
    # Vary the number of electrons in the reservoir and compute the average and std
    # of the smallest singular value of the scrambling map over a set of randomized Hamiltonians
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = QDR.charge_probabilities(grid.total)
    n_qn = 2 * nbr_dots_res + 1
    sv_matrix = zeros(Float64, n_qn, nbr_samples)
    randomized_hams = [get_ham(grid, parameters) for _ in 1:nbr_samples]
    jobs = [(qn_res, j) for j in 1:nbr_samples for qn_res in 0:(n_qn - 1)]

    Threads.@threads for k in eachindex(jobs)
        qn_res, j = jobs[k]
        sv_matrix[qn_res + 1, j] = smallest_sv(
            grid, qn_res, randomized_hams[j], t, measurements)
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
    randomized_hams = [get_ham(grid, parameter_list[i])
                       for i in eachindex(parameter_list), j in 1:nbr_samples]

    smallest_svs = Matrix{Float64}(undef, length(parameter_list), nbr_samples)
    Threads.@threads for idx in CartesianIndices(smallest_svs)
        i, j = Tuple(idx)
        smallest_svs[idx] = smallest_sv(
            grid, qn_res, randomized_hams[i, j], t, measurements)
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