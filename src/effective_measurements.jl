function effective_measurement(op, ρ_reservoir, H_main_qn, H_reservoir, H_total)
    H_main = hilbert_space(keys(H_main_qn), NumberConservation())
    # Is this the best way to extend the Hilbert space? 
    #Could also input the total main Hilbert space and run sector(qn, H) to get the subsystem, but then I loose the option of having ssub H_main with one particle in each dot.

    ρ_reservoir_tot = embed(ρ_reservoir, H_reservoir => H_total; complement=H_main)
    op_rho = partial_trace(op * ρ_reservoir_tot, H_total => H_main; complement=H_reservoir, alg=FermionicHilbertSpaces.FullPartialTraceAlg())
    exp_value(ρ_m) = dot(ρ_m', op_rho)

    function func(ρ_main_qn_vec)
        ρ_main_qn = reshape(ρ_main_qn_vec, dim(H_main_qn), dim(H_main_qn))

        #Pad matrix
        ρ_main = zeros(Complex{Float64}, dim(H_main), dim(H_main))
        index = FermionicHilbertSpaces.indices(H_main_qn, H_main)
        ρ_main[index, index] = ρ_main_qn

        return exp_value(ρ_main)
    end

    lmap = LinearMaps.LinearMap(func, 1, dim(H_main_qn)^2)
    n_eff = reshape(Matrix{Complex{Float64}}(lmap)', dim(H_main_qn), dim(H_main_qn))
    return n_eff
end

effective_measurement(op, ρ_reservoir, qd_system::QuantumDotSystem) =
    effective_measurement(op, ρ_reservoir, qd_system.H_main_qn, qd_system.H_reservoir, qd_system.H_total)