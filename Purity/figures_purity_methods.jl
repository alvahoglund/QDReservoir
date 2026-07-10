includet("estimate_purity.jl")
## =================== Estimate purity from multiple random scrambling maps =====================
Random.seed!(1323)
S_list = [rand_S() for i in 1:10]
H_main = tight_binding_system(2, 1, 1).H_main # temp sys to get H_main
Ω = random_mixed_states(10^5, H_main)
X_list = [QDR.process_complex.((S * Ω)')
          for (S, sys) in S_list]
Y = get_purity(Ω)
σE_list = 10 .^ range(-7, 0, length = 30)

##
mse_mat = get_purity_mse_stats(X_list, Y, σE_list)
mse_mat_from_state = get_purity_from_state_mse_stats(X_list, Ω, σE_list)

##
med = median(mse_mat, dims = 2)
sd = std(mse_mat, dims = 2)
med_from_state = median(mse_mat_from_state, dims = 2)
sd_from_state = std(mse_mat_from_state, dims = 2)

fig = Figure(size = (600, 300))
ax = Axis(fig[1, 1], xlabel = "Noise level (σE)", ylabel = "MSE",
    title = "Purity Estimation MSE vs Noise Level", xscale = log10, yscale = log10)
lines!(ax, σE_list, vec(med), label = "Median MSE")
scatter!(ax, σE_list, vec(med))
band!(ax, σE_list, vec(med), vec(med) .+ vec(sd), alpha = 0.2)
lines!(ax, σE_list, vec(med_from_state), label = "Median MSE from State")
scatter!(ax, σE_list, vec(med_from_state))
band!(ax, σE_list, vec(med_from_state),
    vec(med_from_state) .+ vec(sd_from_state), alpha = 0.2)
axislegend(ax, position = :lt)
fig

save("estimate_purity_multiple_scrambling_maps_loglog.png", fig)