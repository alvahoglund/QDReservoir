using QDReservoir
using LinearAlgebra, Random, CairoMakie, TensorOperations
import QDReservoir as QDR

##
##
# TAB is the operator measuring entanglement in time (2502.12240):
# Measures: im(TAB^2) — purity-like; norm(TAB - TAB') — non-Hermiticity.
function calc_TAB(H, HA, HB, U, ρ::AbstractMatrix)
    HAcomp = FermionicHilbertSpaces.complementary_subsystem(H, HA)
    HBcomp = FermionicHilbertSpaces.complementary_subsystem(H, HB)
    Ut = reshape(U, H, (HBcomp, HB), (HA, HAcomp))
    rho = reshape(ρ, H, (HA, HAcomp))
    @tensoropt TAB[a, b, a', b'] := rho[a, ac1, a1, ac2] * Ut[bc1, b', a1, ac2] *
                                    conj(Ut[bc1, b, a', ac1])
    Hout = tensor_product((HA, HB))
    reshape(TAB, (HA, HB) => Hout)
end
function calc_TAB(H, HA, HB, U, ψ::AbstractVector)
    HAcomp = FermionicHilbertSpaces.complementary_subsystem(H, HA)
    HBcomp = FermionicHilbertSpaces.complementary_subsystem(H, HB)
    Ut = reshape(U, H, (HBcomp, HB), (HA, HAcomp))
    ψAAc = reshape(ψ, H => (HA, HAcomp))
    ψt = U * ψ
    ψtBcB = reshape(ψt, H => (HBcomp, HB))
    @tensoropt TAB[a, b, a', b'] := conj(ψAAc[a, ac]) * ψtBcB[bc, b'] *
                                    conj(Ut[bc, b, a', ac])
    Hout = tensor_product((HA, HB))
    reshape(TAB, (HA, HB) => Hout)
end

function get_info_measures_vs_time(sys, hams, ψmain_state, t_list; dots = :res)
    hams_mat = QDR.matrix_representation_hams(hams, sys)
    ψmain = def_state(ψmain_state, sys.H_main)
    ψres = QDR.ground_state(hams_mat.res, QDR.KrylovKitAlg())
    ψtot = tensor_product((ψmain, ψres), (sys.H_main, sys.H_res) => sys.H_total)
    spatial_label = first ∘ only ∘ keys
    HA = sys.H_main
    HB = if dots isa Tuple{Int, Int}
        B = filter(isequal(dots) ∘ spatial_label,
            FermionicHilbertSpaces.atomic_factors(sys.H_total))
        subregion(B, sys.H_total)
    elseif eltype(dots) <: Tuple{Int, Int}
        B = filter(in(dots) ∘ spatial_label,
            FermionicHilbertSpaces.atomic_factors(sys.H_total))
        subregion(B, sys.H_total)
    elseif dots == :res
        sys.H_res
    else
        error("Invalid dots argument: $dots of type $(typeof(dots)) with eltype $(eltype(dots))")
    end
    HA = subregion(HA, sys.H_total)
    HB = subregion(HB, sys.H_total)
    stack(t_list) do t
        U = QDR.propagator(t, hams_mat.total)
        # TAB = calc_TAB(sys.H_total, HA, HB, U, density_matrix(ψtot))
        TAB = calc_TAB(sys.H_total, HA, HB, U, ψtot)
        ψt = U * ψtot
        1 - 1e-3 < abs(ψt'ψt) < 1 + 1e-3 ||
            warn("State norm deviates from 1: $(abs(ψt'ψt))")
        mut_inf = QDR.mutual_information(ψt, sys.H_total, HA, HB)
        norm(TAB - TAB'), mut_inf
    end
end

##
nbr_dots_main = 2
nbr_dots_res = 3
ψmain_state = singlet
grid = QDR.generate_grid(nbr_dots_main, nbr_dots_res)

seed = 12379
Random.seed!(seed)
hams = hamiltonians(grid)
t_list = range(0, stop = 15, length = 100)
qn_list = 0:4
sys_list = map(qn -> tight_binding_system(grid, qn), qn_list)

using Folds # for multi-threading, can be dangerous
dots = ((2, 2),)
# dots = :res
@profview @time measures_list = map(
    sys -> get_info_measures_vs_time(
        sys, hams, ψmain_state, t_list; dots), sys_list);

##
let savefig = false, pixel_scaling = 0.8
    fig = Figure(size = (600, 500))
    ax_mutual_info = Axis(fig[1, 1], ylabel = "Mutual Information")
    ax_imagitativity = Axis(
        fig[2, 1], xlabel = "Time", ylabel = L"||T_{AB} - T_{AB}^\dagger||")

    for (qn, measures) in zip(qn_list, measures_list)
        lines!(ax_mutual_info, t_list, measures[2, :], label = "$qn")
        lines!(ax_imagitativity, t_list, measures[1, :], label = "$qn")
    end

    header = GridLayout(fig[0, 1])
    Legend(
        header[1, 1], ax_mutual_info, "Electrons in reservoir", orientation = :horizontal)
    Label(
        header[1, 2],
        "N_main = $(nbr_dots_main), N_res = $(nbr_dots_res)\nA = main\nB = $(dots)",
        halign = :left,
        valign = :center
    )
    savefig && save("info_measures_vs_time_t_$(Int(last(t_list)))_dots_$dots.png",
        fig; px_per_unit = pixel_scaling)
    display(fig)
end