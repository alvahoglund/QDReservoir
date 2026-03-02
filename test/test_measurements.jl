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
    sys = tight_binding_system(2, 2, 2)
    state_main = def_state(singlet, sys.H_main, sys.f)
    state_main_vec = reshape(state_main,  dim(sys.H_main)^2, 1)
    state_main_qn = def_state(singlet, sys.H_main_qn, sys.f)
    state_main_qn_vec = reshape(state_main_qn,  dim(sys.H_main_qn)^2, 1)

    pauli_string_list = [(σi, σj) for σi in [QDR.σ0, QDR.σx, QDR.σy, QDR.σz] for σj in [QDR.σ0, QDR.σx, QDR.σy, QDR.σz]]
    for (i,(σi, σj)) in enumerate(pauli_string_list)
        pauli_str_main = QDR.pauli_string(σi, σj, sys.coordinates_main[1], sys.coordinates_main[2], sys.H_main_a, sys.H_main_b, sys.H_main, sys.f)
        pauli_str_main_mat = QDR.pauli_string_matrix(sys.coordinates_main[1], sys.coordinates_main[2], sys.H_main_a, sys.H_main_b, sys.H_main, sys.f)
        pauli_str_main_qn = QDR.pauli_string(σi, σj, sys.coordinates_main[1], sys.coordinates_main[2], sys.H_main_a_qn, sys.H_main_b_qn, sys.H_main_qn, sys.f)
        pauli_str_main_qn_mat = QDR.pauli_string_matrix(sys.coordinates_main[1], sys.coordinates_main[2], sys.H_main_a_qn, sys.H_main_b_qn, sys.H_main_qn, sys.f)

        ev_main = QDR.expectation_value(state_main, pauli_str_main)
        ev_main_qn = QDR.expectation_value(state_main_qn, pauli_str_main_qn)
        ev_main_mat = real(pauli_str_main_mat * state_main_vec)
        ev_main_qn_mat = real(pauli_str_main_qn_mat * state_main_qn_vec)
        @test ev_main ≈ ev_main_qn ≈ ev_main_mat[i] ≈ ev_main_qn_mat[i]
    end
end
