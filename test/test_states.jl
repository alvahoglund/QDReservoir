@testset "Eigenstates" begin
    sys = QDR.tight_binding_system(2, 2, 2)
    hams_symbolic = hamiltonians(sys)
    hamiltonian_total_qn = matrix_representation(hams_symbolic.hamiltonian_total, sys.H_total_qn)
    hamiltonian_total = matrix_representation(hams_symbolic.hamiltonian_total, sys.H_total)    
    
    ψ_arnoldi = ground_state(hamiltonian_total_qn, QDR.ArnoldiAlg())
    ψ_exact = ground_state(hamiltonian_total_qn, QDR.ExactDiagonalizationAlg()) 
    
    ind = indices(sys.qn_total, sys.H_total)
    ψ_qn = ground_state(hamiltonian_total, sys.H_total, sys.qn_total)[ind]
    
    @test(norm(ψ_arnoldi) ≈ norm(ψ_exact) ≈ norm(ψ_qn) ≈ 1.0)
    @test isapprox(abs(ψ_arnoldi' * ψ_exact), 1.0; atol=1e-7)
    @test isapprox(abs(ψ_arnoldi' * ψ_qn), 1.0; atol=1e-7)

    ψ_arnoldi_2 = QDR.eig_state(hamiltonian_total_qn, 2, QDR.ArnoldiAlg())
    ψ_exact_2 = QDR.eig_state(hamiltonian_total_qn, 2, QDR.ExactDiagonalizationAlg())
    ψ_qn_2 = QDR.eig_state(hamiltonian_total, sys.H_total, sys.qn_total, 2)[ind]
    @test isapprox(abs(ψ_arnoldi_2' * ψ_exact_2), 1.0; atol=1e-7)
    @test isapprox(abs(ψ_arnoldi_2' * ψ_qn_2), 1.0; atol=1e-7)
end