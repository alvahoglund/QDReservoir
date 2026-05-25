includet("smallest_singular_value.jl")
using Random, JLD2

##

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
println("Krylov Propagator Alg:")
@time S_krylov = Matrix(scrambling_map(sys, m_ops, ψ_ground, hams.total, t,
    QDR.KrylovPropagatorAlg()))
println("Fraction difference of smallest singular value:")
println((ssv_block - minimum(svdvals(S_krylov))) / ssv_block)

println("=====================")
println("Stepping Krylov Propagator Alg:")
@time S_krylov_step = Matrix(scrambling_map(sys, m_ops, ψ_ground, hams.total, t,
    QDR.SteppingKrylovPropagatorAlg()))
println("Fraction difference of smallest singular value:")
println((ssv_block - minimum(svdvals(S_krylov_step))) / ssv_block)

println("=====================")
println("Diagonalization:")
@time S_diag = Matrix(scrambling_map(sys, m_ops, ψ_ground, hams.total, t,
    QDR.DiagonalizationPropagatorAlg()))
println("Fraction difference of smallest singular value:")
println((ssv_block - minimum(svdvals(S_diag))) / ssv_block)

println("=====================")
println("Adaptive:")
@time S_adaptive = Matrix(scrambling_map(sys, m_ops, ψ_ground, hams.total, t,
    QDR.AdaptivePropagatorAlg()))
println("Fraction difference of smallest singular value:")
println((ssv_block - minimum(svdvals(S_adaptive))) / ssv_block)
