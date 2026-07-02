using QDReservoir, LinearAlgebra, Random, Statistics
import QDReservoir as QDR

# Set BLAS to single-threaded to avoid oversubscription
BLAS.set_num_threads(1)
## ===================== Functions ========================
clean_val(y) = map(x -> abs(x) < 1e-14 ? NaN
                        : x, y)

function smallest_sv(S)
    svd_vals = svdvals(S)
    if length(svd_vals) < 16
        return 0
    end
    return minimum(svd_vals)
end

function smallest_sv(sys, m_ops, hams_matrix, ψ_ground, t)
    S = scrambling_map(
        sys, m_ops, ψ_ground, hams_matrix.total, t)
    return smallest_sv(S)
end

## ===================== Functions for Singular values vs. reservoir electrons ========================

function avg_sv_vs_qn(
        nbr_dots_res, t, nbr_samples, parameters, measurement_func = QDR.charge_probabilities)
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = measurement_func(grid.total)
    n_qn = 2 * nbr_dots_res + 1

    sys_list = map(Base.Fix1(tight_binding_system, grid), 0:(n_qn - 1))
    m_ops_list = map(
        sys -> QDR.matrix_representation_ops(measurements, sys.H_total), sys_list)
    hams_symb = [QDR.hamiltonians(grid, parameters) for _ in 1:nbr_samples]

    hams_mat = Matrix{QDR.Hamiltonians}(undef, n_qn, nbr_samples)
    for idx in CartesianIndices(hams_mat)
        qn_res_idx, j = Tuple(idx)
        hams_mat[qn_res_idx, j] = QDR.matrix_representation_hams(
            hams_symb[j], sys_list[qn_res_idx])
    end
    ψ_res = map(ham_mat -> ground_state(ham_mat.res), hams_mat)
    sv_matrix = zeros(Float64, n_qn, nbr_samples)

    Threads.@threads for idx in CartesianIndices(sv_matrix)
        qn_res_idx, j = Tuple(idx)
        sv_matrix[idx] = smallest_sv(
            sys_list[qn_res_idx], m_ops_list[qn_res_idx],
            hams_mat[qn_res_idx, j], ψ_res[qn_res_idx, j], t)
    end

    mean_sv = vec(mean(sv_matrix, dims = 2))
    std_sv = vec(std(sv_matrix, dims = 2, corrected = false))
    median_sv = vec(median(sv_matrix, dims = 2))

    return (mean = mean_sv, std = std_sv, median = median_sv)
end

function avg_sv_vs_res(nbr_dots_res_list, t, nbr_samples, parameters,
        measurement_func = QDR.charge_probabilities)
    avg_sv_dict = Dict{
        Int, NamedTuple{(:mean, :std, :median),
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
    for nbr_dots_res in nbr_dots_res_list
        println("Calculating for reservoir dots: $(nbr_dots_res)")
        avg_sv_list = avg_sv_vs_qn(
            nbr_dots_res, t, nbr_samples, parameters, measurement_func)
        avg_sv_dict[nbr_dots_res] = avg_sv_list
    end
    return avg_sv_dict
end

## ================= Smallest singular values vs parameter ========================

function avg_sv_vs_param(nbr_dots_res, qn_res, parameter_list, nbr_samples,
        t, measurement_func)
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = measurement_func(grid.total)
    time1 = time()
    sys = tight_binding_system(grid, qn_res)
    m_ops = QDR.matrix_representation_ops(measurements, sys.H_total)

    time2 = time()
    println("Set ops: $(time2 - time1)")

    hams_symb = map(
        Base.Fix1(QDR.hamiltonians, grid), [p for p in parameter_list, _ in 1:nbr_samples])
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

    return (mean = mean_sv, std = std_sv, median = median_sv)
end

function avg_sv_vs_param(reservoir_settings, parameter_list, nbr_samples,
        t, measurement_func = QDR.charge_probabilities)
    avg_sv_dict = Dict{
        Tuple{Int, Int}, NamedTuple{(:mean, :std, :median),
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
    for (nbr_dots_res, qn_res) in reservoir_settings
        println("Calculating for reservoir dots: $(nbr_dots_res), reservoir electrons: $(qn_res)")
        avg_sv_list = avg_sv_vs_param(nbr_dots_res, qn_res, parameter_list, nbr_samples,
            t, measurement_func)
        avg_sv_dict[(nbr_dots_res, qn_res)] = avg_sv_list
    end
    return avg_sv_dict
end

## ================== Smallest sv vs time ==================
function avg_sv_vs_time(
        nbr_dots_res, qn_res, time_list, nbr_samples, parameter_funcs, measurement_func)
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = measurement_func(grid.total)
    sys = tight_binding_system(grid, qn_res)
    m_ops = QDR.matrix_representation_ops(measurements, sys.H_total)
    hams_symb = [QDR.hamiltonians(grid, parameter_funcs) for _ in 1:nbr_samples]
    hams_mat = map(Base.Fix2(QDR.matrix_representation_hams, sys), hams_symb)
    ψ_ground = map(ham_mat -> ground_state(ham_mat.res), hams_mat)

    ssv_matrix = zeros(Float64, length(time_list), nbr_samples)
    Threads.@threads :dynamic for idx in CartesianIndices(ssv_matrix)
        time_idx, sample_idx = Tuple(idx)
        ssv_matrix[idx] = smallest_sv(
            sys, m_ops, hams_mat[sample_idx], ψ_ground[sample_idx], time_list[time_idx])
    end
    mean_sv = vec(mean(ssv_matrix, dims = 2))
    std_sv = vec(std(ssv_matrix, dims = 2, corrected = false))
    median_sv = vec(median(ssv_matrix, dims = 2))
    return (mean = mean_sv, std = std_sv, median = median_sv)
end

function avg_sv_vs_time(
        reservoir_settings, time_list, nbr_samples, parameter_funcs, measurement_func)
    avg_sv_dict = Dict{
        Tuple{Int, Int}, NamedTuple{(:mean, :std, :median),
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
    for (nbr_dots_res, qn_res) in reservoir_settings
        println("Calculating for reservoir dots: $(nbr_dots_res), reservoir electrons: $(qn_res)")
        avg_sv_list = avg_sv_vs_time(nbr_dots_res, qn_res, time_list, nbr_samples,
            parameter_funcs, measurement_func)
        avg_sv_dict[(nbr_dots_res, qn_res)] = avg_sv_list
    end
    return avg_sv_dict
end

## ================== Smallest sv vs hamiltonian multiplexing ==================

function avg_sv_vs_ham_multiplexing(
        nbr_dots_res, qn_res, time_eval, parameter_func, nbr_samples, max_multiplex_hams)
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = QDR.charge_probabilities(grid.total)
    sys = tight_binding_system(grid, qn_res)
    m_ops = QDR.matrix_representation_ops(measurements, sys.H_total)
    scrambling_maps = Matrix{Matrix{ComplexF64}}(undef, max_multiplex_hams, nbr_samples)

    Threads.@threads :dynamic for idx in CartesianIndices(scrambling_maps)
        multiplex_idx, sample_idx = Tuple(idx)
        hams_symb = QDR.hamiltonians(grid, parameter_func)
        hams_mat = QDR.matrix_representation_hams(hams_symb, sys)
        ψ_ground = ground_state(hams_mat.res)
        scrambling_maps[multiplex_idx, sample_idx] = scrambling_map(
            sys, m_ops, ψ_ground, hams_mat.total, time_eval)
    end
    ssv_matrix = zeros(Float64, max_multiplex_hams, nbr_samples)
    Threads.@threads :dynamic for idx in CartesianIndices(ssv_matrix)
        multiplex_idx, sample_idx = Tuple(idx)
        multiplexed_S = vcat(scrambling_maps[1:multiplex_idx, sample_idx]...)
        ssv_matrix[multiplex_idx, sample_idx] = smallest_sv(multiplexed_S)
    end
    mean_sv = vec(mean(ssv_matrix, dims = 2))
    std_sv = vec(std(ssv_matrix, dims = 2, corrected = false))
    median_sv = vec(median(ssv_matrix, dims = 2))
    return (mean = mean_sv, std = std_sv, median = median_sv)
end

function avg_sv_vs_ham_multiplexing(
        reservoir_settings, time_eval, parameter_func, nbr_samples, max_multiplex_hams)
    avg_sv_dict = Dict{
        Tuple{Int, Int}, NamedTuple{(:mean, :std, :median),
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
    for (nbr_dots_res, qn_res) in reservoir_settings
        println("Calculating for reservoir dots: $(nbr_dots_res), reservoir electrons: $(qn_res)")
        avg_sv_list = avg_sv_vs_ham_multiplexing(nbr_dots_res, qn_res, time_eval,
            parameter_func, nbr_samples, max_multiplex_hams)
        avg_sv_dict[(nbr_dots_res, qn_res)] = avg_sv_list
    end
    return avg_sv_dict
end

## ================== Smallest sv vs time multiplexing ==================
function avg_sv_vs_time_multiplexing(
        nbr_dots_res, qn_res, times_multiplexing, parameter_func, nbr_samples)
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = QDR.charge_probabilities(grid.total)
    sys = tight_binding_system(grid, qn_res)
    m_ops = QDR.matrix_representation_ops(measurements, sys.H_total)
    scrambling_maps = Matrix{Matrix{ComplexF64}}(
        undef, length(times_multiplexing), nbr_samples)

    hams_symb_list = [QDR.hamiltonians(grid, parameter_func) for i in 1:nbr_samples]
    hams_mat_list = [QDR.matrix_representation_hams(ham_symb, sys)
                     for ham_symb in hams_symb_list]
    ψ_ground_list = [ground_state(hams_mat.res) for hams_mat in hams_mat_list]

    Threads.@threads :dynamic for idx in CartesianIndices(scrambling_maps)
        time_idx, sample_idx = Tuple(idx)

        scrambling_maps[
            time_idx, sample_idx] = scrambling_map(
            sys, m_ops, ψ_ground_list[sample_idx],
            hams_mat_list[sample_idx].total, times_multiplexing[time_idx])
    end

    ssv_matrix = zeros(Float64, length(times_multiplexing), nbr_samples)
    n_times = length(times_multiplexing)
    Threads.@threads :dynamic for idx in CartesianIndices(ssv_matrix)
        time_idx, sample_idx = Tuple(idx)
        sel = time_idx == 1 ? [1] :
              round.(Int, range(1, n_times, length = time_idx))
        multiplexed_S = vcat(scrambling_maps[sel, sample_idx]...)
        ssv_matrix[time_idx, sample_idx] = smallest_sv(multiplexed_S)
    end
    mean_sv = vec(mean(ssv_matrix, dims = 2))
    std_sv = vec(std(ssv_matrix, dims = 2, corrected = false))
    median_sv = vec(median(ssv_matrix, dims = 2))
    return (mean = mean_sv, std = std_sv, median = median_sv)
end

function avg_sv_vs_time_multiplexing(
        reservoir_settings, times_multiplexing, parameter_func, nbr_samples)
    avg_sv_dict = Dict{
        Tuple{Int, Int}, NamedTuple{(:mean, :std, :median),
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
    for (nbr_dots_res, qn_res) in reservoir_settings
        println("Calculating for reservoir dots: $(nbr_dots_res), reservoir electrons: $(qn_res)")
        avg_sv_list = avg_sv_vs_time_multiplexing(nbr_dots_res, qn_res,
            times_multiplexing, parameter_func, nbr_samples)
        avg_sv_dict[(nbr_dots_res, qn_res)] = avg_sv_list
    end
    return avg_sv_dict
end

## ================== Smallest sv vs multiplexing with fixed time ==================

function avg_sv_vs_multiplexing_fixed_time(
        nbr_dots_res, qn_res, max_multiplexing, parameter_func, nbr_samples, time_eval)
    grid = QDR.generate_grid(2, nbr_dots_res)
    measurements = QDR.charge_probabilities(grid.total)
    sys = tight_binding_system(grid, qn_res)
    m_ops = QDR.matrix_representation_ops(measurements, sys.H_total)
    scrambling_maps = Vector{Matrix{ComplexF64}}(undef, nbr_samples)
    Threads.@threads :dynamic for idx in eachindex(scrambling_maps)
        hams_symb = QDR.hamiltonians(grid, parameter_func)
        hams_mat = QDR.matrix_representation_hams(hams_symb, sys)
        ψ_ground = ground_state(hams_mat.res)
        scrambling_maps[idx] = scrambling_map(
            sys, m_ops, ψ_ground, hams_mat.total, time_eval)
    end
    ssv_matrix = zeros(Float64, max_multiplexing, nbr_samples)
    for idx in CartesianIndices(ssv_matrix)
        multiplex_idx, sample_idx = Tuple(idx)
        multiplexed_S = vcat(scrambling_maps[1:multiplex_idx, sample_idx]...)
        ssv_matrix[idx] = smallest_sv(multiplexed_S)
    end
    mean_sv = vec(mean(ssv_matrix, dims = 2))
    std_sv = vec(std(ssv_matrix, dims = 2, corrected = false))
    median_sv = vec(median(ssv_matrix, dims = 2))
    return (mean = mean_sv, std = std_sv, median = median_sv)
end

function avg_sv_vs_multiplexing_fixed_time(
        reservoir_settings, max_multiplexing, parameter_func, nbr_samples, time_eval)
    avg_sv_dict = Dict{
        Tuple{Int, Int}, NamedTuple{(:mean, :std, :median),
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
    for (nbr_dots_res, qn_res) in reservoir_settings
        println("Calculating for reservoir dots: $(nbr_dots_res), reservoir electrons: $(qn_res)")
        avg_sv_list = avg_sv_vs_multiplexing_fixed_time(nbr_dots_res, qn_res,
            max_multiplexing, parameter_func, nbr_samples, time_eval)
        avg_sv_dict[(nbr_dots_res, qn_res)] = avg_sv_list
    end
    return avg_sv_dict
end