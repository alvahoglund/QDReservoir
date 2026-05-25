
function make_test_system(; u_intra = 10.0, t = 1.0, t_so = 0.1, u_inter = 1.0)
    qd_system = tight_binding_system(2, 3, 1)
    dot_params_main = set_dot_params(() -> 0.5, () -> [0, 0, 1], () -> u_intra,
        qd_system.grids.main)
    dot_params_res = set_dot_params(() -> 0.5, () -> [0, 0, 1], () -> u_intra,
        qd_system.grids.res)
    interaction_params = set_interaction_params(
        () -> t, () -> t_so, () -> u_inter, qd_system.grids.total)
    hams = hamiltonians(
        qd_system.grids, dot_params_main, dot_params_res, interaction_params)
    ham_total = matrix_representation(hams.total, qd_system.H_total)
    ham_res = matrix_representation(hams.res, qd_system.H_res)
    ψ_res = QDR.eig_state(ham_res, 2)
    measurements = charge_measurements(qd_system)
    return qd_system, measurements, ψ_res, ham_total
end

function test_algorithms_agree(qd_system, measurements, ψ_res, ham_total, t; atol = 1e-4)
    S_diag = scrambling_map(qd_system, measurements, ψ_res, ham_total, t,
        QDR.DiagonalizationPropagatorAlg())
    S_stepping = scrambling_map(qd_system, measurements, ψ_res, ham_total, t,
        QDR.SteppingKrylovPropagatorAlg())
    S_adaptive = scrambling_map(qd_system, measurements, ψ_res, ham_total, t,
        QDR.AdaptivePropagatorAlg())
    @test S_stepping≈S_diag atol=atol
    @test S_adaptive≈S_diag atol=atol
end

@testset "Scrambling map algorithms agree — default parameters, large t" begin
    qd_system, measurements, ψ_res, ham_total = make_test_system()
    test_algorithms_agree(qd_system, measurements, ψ_res, ham_total, 1000.0)
end

@testset "Scrambling map algorithms agree — large U_intra (large spectral radius)" begin
    qd_system, measurements, ψ_res, ham_total = make_test_system(u_intra = 100.0)
    test_algorithms_agree(qd_system, measurements, ψ_res, ham_total, 1000.0)
end

@testset "Scrambling map algorithms agree — large t, multiple times" begin
    qd_system, measurements, ψ_res, ham_total = make_test_system()
    test_algorithms_agree(qd_system, measurements, ψ_res, ham_total, [200.0, 1000.0])
end

@testset "Scrambling map algorithms agree — large U_intra, multiple times" begin
    qd_system, measurements, ψ_res, ham_total = make_test_system(u_intra = 100.0)
    test_algorithms_agree(qd_system, measurements, ψ_res, ham_total, [200.0, 1000.0])
end
