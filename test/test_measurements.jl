@testset "Charge Probabilities" begin
    # Test probability functions sum to 1
    coordinate = (1, 1)
    @fermions f
    H = hilbert_space([(coordinate, :↑), (coordinate, :↓)], NumberConservation())

    p0_val = matrix_representation(QDR.p0(coordinate, f), H)
    p1_val = matrix_representation(QDR.p1(coordinate, f), H)
    p2_val = matrix_representation(QDR.p2(coordinate, f), H)

    @test p0_val + p1_val + p2_val ≈ I
end

@testset "Expectation Value of Charge Measurement" begin
    coordinate = (1, 1)
    @fermions f
    H = hilbert_space([(coordinate, :↑), (coordinate, :↓)], NumberConservation(1))

    state = [0.5 0.0; 0.0 0.5]
    p1_op = matrix_representation(QDR.p1(coordinate, f), H)

    ev = expectation_value(state, p1_op)
    @test ev ≈ 1.0
end

@testset "Pauli strings" begin
    import QDReservoir as QDR
    sys = tight_binding_system(2, 2, 2)
    paulis = QDR.pauli_strings(sys.Hs_main, sys.H_main)

    @test length(paulis) == 16
    ops = collect(values(paulis))
    @test map(ops -> dot(ops...), Base.product(ops, ops)) ≈ 4 * I
end
