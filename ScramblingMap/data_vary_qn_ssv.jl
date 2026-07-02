includet("smallest_singular_value.jl")
using Random, JLD2

nbr_samples = 10
time_eval = [30, 60]

ham_params = QDR.random_param_functions()

function charge_probabilities_4(grid)
    vcat(QDR.zero_charge_probabilities(grid[1:4]),
        QDR.single_charge_probabilities(grid[1:4]),
        QDR.double_charge_probabilities(grid[1:4]))
end

nbr_dots_res_list = [2, 3, 4, 5]
ssv_dict = avg_sv_vs_res(
    nbr_dots_res_list, time_eval, nbr_samples, ham_params, charge_probabilities_4)

jldsave("ScramblingMap/data_vary_qn_ssv/data_vary_qn_ssv_measure_all_dots_2dots.jld2";
    ssv_dict, nbr_dots_res_list, time_eval, nbr_samples)
