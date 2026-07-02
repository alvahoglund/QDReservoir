includet("smallest_singular_value.jl")
using Random, JLD2
##
nbr_samples = 10
parameter_funcs = QDR.random_param_functions()
measurement_func = QDR.charge_probabilities
## ================== Vary time without multiplexing ==================
seed = 42899
Random.seed!(seed)

reservoir_settings = [
    (3, 3), (6, 3)]

time_list = [[1 * i] for i in range(0.001, 100, length = 100)]
ssv_dict_time = avg_sv_vs_time(
    reservoir_settings, time_list, nbr_samples, parameter_funcs, measurement_func)
jldsave("SmallestSingularValue/data_vary_time_ssv/avg_sv_dict_time.jld2";
    ssv_dict_time, time_list, nbr_samples)

## ================== Vary time with multiplexing ==================
seed = 42899
Random.seed!(seed)
reservoir_settings = [
    (3, 3), (6, 3)]

time_list = [[1 * i, 2 * i] for i in range(0.001, 100, length = 100)]
ssv_dict_time_multiplex = avg_sv_vs_time(
    reservoir_settings, time_list, nbr_samples, parameter_funcs, measurement_func)
jldsave("SmallestSingularValue/data_vary_time_ssv/avg_sv_dict_time_multiplex.jld2";
    ssv_dict_time_multiplex, time_list, nbr_samples)

## ================== Vary multiplexing with fixed time ==================
