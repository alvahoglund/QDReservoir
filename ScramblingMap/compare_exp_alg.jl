includet("smallest_singular_value.jl")
using CairoMakie, Random, JLD2

##
function compare_methods(sys, params, t)
    hams = QDR.matrix_representation_hams(get_ham(sys.grids, params), sys)
    m_ops = QDR.matrix_representation_ops(
        QDR.charge_measurements(sys.grids.total), sys.H_total)
    ψ_ground = ground_state(hams.res)

    println("=======================")
    println("Exact solution:")
    @time S_block = scrambling_map(sys, m_ops, ψ_ground, hams.total, t,
        QDR.BlockPropagatorAlg())
    ssv_block = minimum(svdvals(S_block))

    println("=======================")
    println("PureStatePropagatorAlg:")
    @time S_krylov = Matrix(scrambling_map(sys, m_ops, ψ_ground, hams.total, t,
        QDR.PureStatePropagatorAlg()))
    println("Fraction difference of smallest singular value:")
    println((ssv_block - minimum(svdvals(S_krylov))) / ssv_block)

    println("=====================")
    println("Adaptive stepping:")
    @time S_krylov_step = Matrix(scrambling_map(sys, m_ops, ψ_ground, hams.total, t,
        QDR.PureStateSteppingPropagatorAlg()))
    println("Fraction difference of smallest singular value:")
    println((ssv_block - minimum(svdvals(S_krylov_step))) / ssv_block)
end

seed = 2
Random.seed!(seed)

sys = QDR.tight_binding_system(2, 5, 3)
params = (
    ϵ_func_main = () -> 0.5,
    ϵ_func_res = () -> rand(),
    ϵb_func = () -> [0, 0, 1],
    u_intra_func = () -> 1 * (10 + rand()), # <-- Change to vary spectral bandwidth
    t_func = () -> rand(),
    t_so_func = () -> 0.1 * rand(),
    u_inter_func = () -> rand()
)
t = 100

compare_methods(sys, params, t)
