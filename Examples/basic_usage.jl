using QDReservoir
import QDReservoir as QDR
## Define a system
nbr_dots_main = 2
nbr_dots_res = 2
qn_res = 2
sys = QDR.tight_binding_system(nbr_dots_main, nbr_dots_res, qn_res)
hams = QDR.matrix_representation_hams(hamiltonians(sys.grids), sys)

## SET STATE
ψ_main = def_state(singlet, sys.H_main)
ψ_res = QDR.ground_state(hams.res)
ψ_tot = generalized_kron((ψ_main, ψ_res), (sys.H_main, sys.H_res) => sys.H_total)

## Measurements
m_list = QDR.charge_probabilities(sys)
t = 10
S = scrambling_map(sys, m_list, ψ_res, hams.total, t)
S * reshape(density_matrix(ψ_main), 16, 1)