
function effective_measurement(op, ρ_reservoir, H_main, H_reservoir, H_total)
    ρ_res_extend = embed(ρ_reservoir, H_reservoir => H_total)
    partial_trace(op*ρ_res_extend, H_total => H_main, alg=FermionicHilbertSpaces.FullPartialTraceAlg())
end

effective_measurement(op, ρ_reservoir, qd_system::QuantumDotSystem) =
    effective_measurement(op, ρ_reservoir, qd_system.H_main_qn, qd_system.H_reservoir_qn, qd_system.H_total_qn)