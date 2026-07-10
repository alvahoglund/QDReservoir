includet("smallest_singular_value.jl")
using Random, JLD2
##
nbr_samples = 10
parameter_funcs = QDR.random_param_functions()
measurement_func = QDR.charge_probabilities
time_eval = [50]

## ================== Vary Nothing Multiplexing ==================
seed = 42899
Random.seed!(seed)
max_multiplexing = 100

reservoir_settings = [
    (6, 3)]
sv_dict_nothing_multiplexing = avg_sv_vs_multiplexing_fixed_time(
    reservoir_settings, max_multiplexing, parameter_funcs, nbr_samples, time_eval)

jldsave(
    "SmallestSingularValue/data_hamiltonian_multiplex_ssv/data_nothing_multiplex_ssv.jld2";
    sv_dict_nothing_multiplexing, reservoir_settings, max_multiplexing, nbr_samples, time_eval)
## ================== Vary Hamiltonian Multiplexing ==================
max_multiplex_hams = 100
seed = 42899
Random.seed!(seed)
reservoir_settings = [
    (3, 3), (6, 3)]
sv_dict_ham_multiplexing = avg_sv_vs_ham_multiplexing(
    reservoir_settings, time_eval,
    parameter_funcs, nbr_samples, max_multiplex_hams)

jldsave("SmallestSingularValue/data_hamiltonian_multiplex_ssv/data_ham_multiplex_ssv.jld2";
    sv_dict_ham_multiplexing, reservoir_settings, time_eval, nbr_samples, max_multiplex_hams)

## ================== Vary time with multiplexing ==================
seed = 42899
Random.seed!(seed)
reservoir_settings = [
    (3, 3), (6, 3)]
times_multiplexing = rand(1:200, 100)

ssv_dict_time_multiplex = avg_sv_vs_time_multiplexing(
    reservoir_settings, times_multiplexing, parameter_funcs, nbr_samples)

jldsave("SmallestSingularValue/data_time_multiplex_ssv/data_time_multiplex_ssv.jld2";
    ssv_dict_time_multiplex, times_multiplexing, nbr_samples)
