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
    paulis = QDR.pauli_strings(sys.Hs_main, sys.H_main)

    @test length(paulis) == 16
    ops = collect(values(paulis))
    @test map(ops -> dot(ops...), Base.product(ops, ops)) ≈ 4 * I
end

@testset "Spin measurements" begin
    using StatsBase
    sys = tight_binding_system(2, 2, 2)

    #Single spin
    single_spin = QDR.random_state(sys.Hs_main[1])
    s2_op = QDR.total_spin_op([sys.grid.main[1]], sys.f, sys.Hs_main[1])
    s2_exp = QDR.expectation_value(single_spin, s2_op)
    @test s2_exp ≈ 3 / 4
    @test QDR.s_from_s2(s2_exp) ≈ 1 / 2

    #Singlet and triplets
    s2_func(state) = expectation_value(def_state(state, sys.H_main, sys.f), QDR.total_spin_op(sys.grid.main, sys.f, sys.H_main))
    s_func(state) = QDR.s_from_s2(expectation_value(def_state(state, sys.H_main, sys.f), QDR.total_spin_op(sys.grid.main, sys.f, sys.H_main)))
    @test s2_func(QDR.triplet_0) ≈ s2_func(QDR.triplet_minus) ≈ s2_func(QDR.triplet_plus) ≈ 2
    @test s_func(QDR.triplet_0) ≈ s_func(QDR.triplet_minus) ≈ s_func(QDR.triplet_plus) ≈ 1
    @test s2_func(QDR.singlet) ≈ 0
    @test s_func(QDR.singlet) ≈ 0

    #Eigenvalues of spin operator
    function allowed_spins_half(qn)
        iseven(qn) ? (qn/2:-1:0) : (qn/2:-1:1/2)
    end
    function allowed_spins(nbr_dots, qn)
        spins = Set{Float64}()
        for d in 0:floor(Int, qn / 2)
            s = qn - 2d
            if s + d <= nbr_dots
                union!(spins, allowed_spins_half(s))
            end
        end
        return sort(collect(spins), rev=true)
    end

    S2_list(nbr_dots, qn) = [s * (s + 1) for s in allowed_spins(nbr_dots, qn)]

    for nbr_res in 0:3
        for qn in 0:nbr_res*2
            sys = tight_binding_system(2, nbr_res, qn)
            s2_op = QDR.total_spin_op(sys.grid.total, sys.f, sys.H_total)
            vals = round.(eigen(Matrix(s2_op)).values, digits=4)

            S2_exp = S2_list(nbr_res + 2, qn + 2)
            @test sort!(unique(abs.(vals))) ≈ sort!(S2_exp)
        end
    end
end