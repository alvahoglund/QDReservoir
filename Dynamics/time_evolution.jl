using QDReservoir
using LinearAlgebra, Statistics, CairoMakie, Distributions, Random
import QDReservoir as QDR

##
function von_neumann(ρ)
    λ = real(eigen(ρ).values)
    λ = λ[λ .> 0]
    S = -sum(λ .* log.(λ))
    return S
end

function mutual_information(ψ, sys)
    ρmain = partial_trace(density_matrix(ψ), sys.H_total => sys.H_main)
    von_neumann(ρmain)
end

function get_mutual_information_vs_time(sys, hams, ψmain_state, t_list)
    hams = QDR.matrix_representation_hams(hams, sys)
    ψmain = def_state(ψmain_state, sys.H_main)
    ψres = QDR.ground_state(hams.res)
    ψtot = generalized_kron((ψmain, ψres), (sys.H_main, sys.H_res) => sys.H_total)
    propagators = map(Base.Fix2(QDR.propagator, hams.total), t_list)
    ψt = map(p -> p * ψtot, propagators)
    mutual_informations = map(Base.Fix2(mutual_information, sys), ψt)
    return mutual_informations
end

function plot_mutual_information(t_list, mutual_informations_list, qn_list)
    fig = Figure(size = (600, 400))
    ax = Axis(fig[1, 1], xlabel = "Time", ylabel = "Mutual Information")
    for (qn, mutual_informations) in zip(qn_list, mutual_informations_list)
        lines!(ax, t_list, mutual_informations, label = "$qn")
    end
    axislegend(ax, "Electrons in reservoir", position = :lt, orientation = :horizontal)
    display(fig)
end

##
nbr_dots_main = 2
nbr_dots_res = 2
ψmain_state = singlet
grid = QDR.generate_grid(nbr_dots_main, nbr_dots_res)

seed = 12379
Random.seed!(seed)
hams = hamiltonians(grid)
t_list = range(0, stop = 15, length = 200)
qn_list = 0:4
sys_list = map(qn -> tight_binding_system(grid, qn), qn_list)

mutual_informations_list = map(
    sys -> get_mutual_information_vs_time(sys, hams, ψmain_state, t_list), sys_list)

plot_mutual_information(t_list, mutual_informations_list, qn_list)