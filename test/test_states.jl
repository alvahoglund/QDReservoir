@testset "Eigenstates" begin
    Nmain = 2
    Nres = 2
    qn_res = 2
    qn_total = 2 + 2
    sys = QDR.tight_binding_system(Nmain, Nres, qn_res)
    hams_symbolic = hamiltonians(sys)
    hamiltonian_total = matrix_representation(hams_symbolic.hamiltonian_total, sys.H_total)

    ψ_arnoldi = ground_state(hamiltonian_total, QDR.ArnoldiAlg())
    ψ_exact = ground_state(hamiltonian_total, QDR.ExactDiagonalizationAlg())

    @test(norm(ψ_arnoldi) ≈ norm(ψ_exact) ≈ 1.0)
    @test isapprox(abs(ψ_arnoldi' * ψ_exact), 1.0; atol=1e-7)
end