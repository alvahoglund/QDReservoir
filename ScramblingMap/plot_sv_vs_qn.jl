includet("smallest_singular_value.jl")

## ===================== Parameters and system generation ========================
##All parameters
seed = 29084
Random.seed!(seed)

parameters1 = (
    ϵ_func_main = () -> 0.5,
    ϵ_func_res = () -> rand(),
    ϵb_func = () -> [0, 0, 1],
    u_intra_func = () -> 10 + rand(),
    t_func = () -> rand(),
    t_so_func = () -> 0.1 * rand(),
    u_inter_func = () -> rand()
)

nbr_dots_res_list = [2, 3, 4, 5, 6]
t = [100, 200]
nbr_samples = 10

avg_smallest_sv_dict_1 = avg_smallest_sv(nbr_dots_res_list, t, nbr_samples, parameters1)

title = L"H = H_ϵ + H_B + H_{U}^{\text{intra}} +H_{U}^{\text{inter}} + H_t + H_{SO}"
plot_avg_sv_vs_qn(avg_smallest_sv_dict_1, title)
##Removed Uinter
seed = 42879
Random.seed!(seed)

parameters2 = (
    ϵ_func_main = () -> 0.5,
    ϵ_func_res = () -> rand(),
    ϵb_func = () -> [0, 0, 1],
    u_intra_func = () -> 10 + rand(),
    t_func = () -> rand(),
    t_so_func = () -> 0.1 * rand(),
    u_inter_func = () -> 0
)

nbr_dots_res_list = [2, 3, 4, 5, 6]
t = [100, 200]
nbr_samples = 10

avg_smallest_sv_dict_2 = avg_smallest_sv(nbr_dots_res_list, t, nbr_samples, parameters2)
title = L"H = H_ϵ + H_B + H_{U}^{\text{intra}} + H_t + H_{SO}"
plot_avg_sv_vs_qn(avg_smallest_sv_dict_2, title)

##Removed ϵb
seed = 298478
Random.seed!(seed)

parameters3 = (
    ϵ_func_main = () -> 0.5,
    ϵ_func_res = () -> rand(),
    ϵb_func = () -> [0, 0, 10^(-5)],
    u_intra_func = () -> 10 + rand(),
    t_func = () -> rand(),
    t_so_func = () -> 0.1 * rand(),
    u_inter_func = () -> rand()
)

nbr_dots_res_list = [2, 3, 4, 5, 6]
t = [100, 200]
nbr_samples = 10

avg_smallest_sv_dict_3 = avg_smallest_sv(nbr_dots_res_list, t, nbr_samples, parameters3)
title = L"H = H_ϵ + H_{U}^{\text{intra}} + H_{U}^{\text{inter}} + H_t + H_{SO}"
plot_avg_sv_vs_qn(avg_smallest_sv_dict_3, title)

##Removed SO
seed = 12830
Random.seed!(seed)

parameters4 = (
    ϵ_func_main = () -> 0.5,
    ϵ_func_res = () -> rand(),
    ϵb_func = () -> [0, 0, 1],
    u_intra_func = () -> 10 + rand(),
    t_func = () -> rand(),
    t_so_func = () -> 0,
    u_inter_func = () -> rand()
)

nbr_dots_res_list = [2, 3, 4, 5, 6]
t = [100, 200]
nbr_samples = 10

avg_smallest_sv_dict_4 = avg_smallest_sv(nbr_dots_res_list, t, nbr_samples, parameters4)
title = L"H = H_ϵ + H_B + H_{U}^{\text{intra}} + H_{U}^{\text{inter}} + H_t"
plot_avg_sv_vs_qn(avg_smallest_sv_dict_4, title)
