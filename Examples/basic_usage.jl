using QDReservoir
import QDReservoir as QDR
## Define a system
nbr_dots_main = 2
nbr_dots_res = 2
qn_res = 2
sys = QDR.tight_binding_system(nbr_dots_main, nbr_dots_res, qn_res)
hams = QDR.matrix_representation_hams(hamiltonians(sys), sys)

## SET STATE
ρ_main = def_state(singlet, sys.H_main)
ψ_res = QDR.ground_state(hams.hamiltonian_reservoir)
ρ_res = ψ_res * ψ_res'
ρ_tot = tensor_product((ρ_main, ρ_res), (sys.H_main, sys.H_reservoir) => sys.H_total)

## Measurements
m_list = QDR.charge_probabilities(sys)
t = 10
S = scrambling_map(sys, m_list, ψ_res, hams.hamiltonian_total, t)
S * reshape(ρ_main, 16, 1)