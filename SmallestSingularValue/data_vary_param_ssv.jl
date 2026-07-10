includet("smallest_singular_value.jl")
using Random, JLD2
##
nbr_samples = 10
time_eval = [10, 20]
reservoir_settings = [
    (3, 3), (6, 3)]

## ================== Vary SO ==================
seed = 42899
Random.seed!(seed)

tso_range = range(0, 5, length = 20)
parameters_list_tso = [QDR.random_param_functions(t_so = tso) for tso in tso_range]

ssv_dict_so = avg_sv_vs_param(
    reservoir_settings, parameters_list_tso, nbr_samples, time_eval)
jldsave("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_so.jld2";
    ssv_dict_so, tso_range, nbr_samples, time_eval)

## ================== Vary t ===================
seed = 42899
Random.seed!(seed)

t_range = range(0, 5, length = 20)
parameters_list_t = [QDR.random_param_functions(t = t) for t in t_range]
ssv_dict_t = avg_sv_vs_param(reservoir_settings, parameters_list_t, nbr_samples, time_eval)
jldsave("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_t.jld2";
    ssv_dict_t, t_range, nbr_samples, time_eval)

## ================== Vary ϵb ==================
seed = 42899
Random.seed!(seed)

ϵb_range = [[0, 0, b] for b in range(0, 5, length = 20)]
parameters_list_ϵb = [QDR.random_param_functions(ϵb = ϵb) for ϵb in ϵb_range]
ssv_dict_eb = avg_sv_vs_param(
    reservoir_settings, parameters_list_ϵb, nbr_samples, time_eval)
jldsave("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_eb.jld2";
    ssv_dict_eb, ϵb_range, nbr_samples, time_eval)

## ================== Vary u_intra ==================
seed = 42899
Random.seed!(seed)
u_intra_range = range(0, 5, length = 20)
parameters_list_uintra = [QDR.random_param_functions(u_intra = u_intra)
                          for u_intra in u_intra_range]

ssv_dict_uintra = avg_sv_vs_param(
    reservoir_settings, parameters_list_uintra, nbr_samples, time_eval)
jldsave("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_uintra.jld2";
    ssv_dict_uintra, u_intra_range, nbr_samples, time_eval)

## ================== Vary u_inter ==================
seed = 42899
Random.seed!(seed)

u_inter_range = range(0, 5, length = 20)
parameters_list_uinter = [QDR.random_param_functions(u_inter = u_inter)
                          for u_inter in u_inter_range]

ssv_dict_uinter = avg_sv_vs_param(
    reservoir_settings, parameters_list_uinter, nbr_samples, time_eval)
jldsave("SmallestSingularValue/data_vary_param_ssv/avg_sv_dict_uinter.jld2";
    ssv_dict_uinter, u_inter_range, nbr_samples, time_eval)