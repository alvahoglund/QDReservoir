includet("compare_tasks.jl")
includet("plots_compare_tasks.jl")
using JLD2, CairoMakie

## ============== Generate states ================

S = load("DefaultSystems/scrambling_map_A.jld2", "S")
sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")

nbr_sep_states = 10^4
nbr_ent_states = 10^4
nbr_train = (nbr_sep_states + nbr_ent_states) ÷ 2

Ω_sep_linear, Ω_ent_linear = linear_ew_states(sys, nbr_sep_states, nbr_ent_states)
Ω_sep_nonlinear, Ω_ent_nonlinear = nonlinear_ew_states(
    sys, nbr_sep_states, nbr_ent_states)
Ω_purity = random_mixed_states(nbr_sep_states + nbr_ent_states, sys)
Ω_spin = stack(vec(QDR.hilbert_schmidt_ensemble(sys.H_main))
for i in 1:(nbr_sep_states + nbr_ent_states))

## ============== Performance of different tasks ================
σE = 0.1
Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)

linear_ew_result = linear_ew_performance(S, Ω_sep_linear, Ω_ent_linear, σE)
nonlinear_ew_result = nonlinear_ew_performance(
    S, Ω_sep_nonlinear, Ω_ent_nonlinear, σE)
purity_mse = purity_prediction_mse(S, Ω_purity, σE)
spin_mse = spin_prediction_mse(S, Ω_spin, Pm, σE)

## ============== Performance for varying noise levels ================
S = load("DefaultSystems/scrambling_map_A.jld2", "S")
sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")
Pm, Pm_dict = QDR.pauli_matrix(sys.Hs_main, sys.H_main)

σE_list = 10 .^ range(-5, 0, length = 25)
@time linear_ew_results = vcat([linear_ew_performance(S, Ω_sep_linear, Ω_ent_linear, σE)
                                for σE in σE_list]...)
@time nonlinear_ew_results = vcat([nonlinear_ew_performance(
                                       S, Ω_sep_nonlinear, Ω_ent_nonlinear, σE)
                                   for σE in σE_list]...)
@time purity_mse_list = vcat([purity_prediction_mse(S, Ω_purity, σE) for σE in σE_list]...)
@time spin_mse_list = vcat([spin_prediction_mse(S, Ω_spin, Pm, σE) for σE in σE_list]...)

##
svdvals_S = svdvals(S)
fig = Figure(size = (600, 300))
ax1 = Axis(
    fig[1, 1], xlabel = "Noise level (σE)", ylabel = "Fraction Incorrectly Classified/MSE",
    title = "Performance of different tasks", xscale = log10)
lines!(ax1, σE_list, linear_ew_results, label = "Linear Entanglement \nDetection")
lines!(ax1, σE_list, nonlinear_ew_results, label = "Nonlinear Entanglement \nDetection")
lines!(ax1, σE_list, purity_mse_list ./ maximum(purity_mse_list),
    label = "Purity Estimation")
lines!(ax1, σE_list, spin_mse_list ./ maximum(spin_mse_list),
    label = "Average Spin Estimation")
#vlines!(ax1, svdvals_S, linestyle = :dash, color = :grey,
#    label = "σS")
vlines!(ax1, [10^(-5), 10^(-3), 10^(-1.5)], linestyle = :dash, color = :grey)
Legend(fig[1, 2], ax1)

fig
#save("Figures/compare_tasks_performance.png", fig)

## ============== Performance for varying reservoirs ================
nbr_S = 100
sys_list = [randomize_system() for _ in 1:nbr_S]
S_list = Vector{Any}(undef, length(sys_list))
Threads.@threads for i in eachindex(sys_list)
    S_list[i] = get_scrambling_map(sys_list[i]...)
end
ssv_list = [minimum(svdvals(S)) for S in S_list]
# Sort reservoirs by their smallest singular value
sorted_indices = sortperm(ssv_list)
ssv_list = ssv_list[sorted_indices]
S_list = S_list[sorted_indices]
sys_list = sys_list[sorted_indices]

σE_list = [10^-6, 10^-5, 10^-4, 10^-3, 10^-2]
linear_ew_results, nonlinear_ew_results, purity_mse_list, spin_mse_list = get_performances_matrix(
    S_list, Ω_sep_linear, Ω_ent_linear, Ω_sep_nonlinear,
    Ω_ent_nonlinear, Ω_purity, Ω_spin, Pm, σE_list)
#jldsave("EntanglementWitness\\compare_tasks_data\\compare_tasks_data2.jld2";
#    linear_ew_results, nonlinear_ew_results, purity_mse_list, spin_mse_list, ssv_list, σE_list)

## ============== Plotting ================
linear_ew_results, nonlinear_ew_results, purity_mse_list, spin_mse_list, ssv_list, σE_list = jldopen(
    "CompareTasks\\compare_tasks_data\\compare_tasks_data2.jld2", "r") do file
    linear_ew_results = read(file, "linear_ew_results")
    nonlinear_ew_results = read(file, "nonlinear_ew_results")
    purity_mse_list = read(file, "purity_mse_list")
    spin_mse_list = read(file, "spin_mse_list")
    ssv_list = read(file, "ssv_list")
    σE_list = read(file, "σE_list")
    return linear_ew_results,
    nonlinear_ew_results, purity_mse_list, spin_mse_list, ssv_list, σE_list
end
plot_performance_vs_ssv(ssv_list, linear_ew_results, nonlinear_ew_results,
    purity_mse_list, spin_mse_list, σE_list)
plot_lines(ssv_list, linear_ew_results, nonlinear_ew_results,
    purity_mse_list, spin_mse_list, σE_list; n_bins = 10,
    yscale = Makie.Symlog10(1e-3), include_bands = true)
plot_heatmap(ssv_list, linear_ew_results, nonlinear_ew_results,
    purity_mse_list, spin_mse_list, σE_list; n_bins = 25)
