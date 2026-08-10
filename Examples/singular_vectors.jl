using QDReservoir
import QDReservoir as QDR
## Define a system
nbr_dots_main = 2
nbr_dots_res = 6
qn_res = 0
sys = QDR.tight_binding_system(nbr_dots_main, nbr_dots_res, qn_res)

ϵ_func() = rand()
ϵb_func() = [0, 0, 1 * rand()]
u_intra_func() = 2 * rand()
t_func() = 10
t_so_func() = 10
u_inter_func() = 1 * rand()
param = QDR.ParamFunctions(
    ϵ_func_main = ϵ_func, ϵ_func_res = ϵ_func, ϵb_func = ϵb_func, u_intra_func = u_intra_func,
    t_func = t_func, t_so_func = t_so_func, u_inter_func = u_inter_func)

hams = QDR.matrix_representation_hams(
    QDR.hamiltonians(sys.grids, param), sys)

#hams = QDR.matrix_representation_hams(hamiltonians(sys.grids), sys)

## SET STATE
ψ_main = def_state(singlet, sys.H_main)
ψ_res = QDR.ground_state(hams.res)
ψ_tot = generalized_kron((ψ_main, ψ_res), (sys.H_main, sys.H_res) => sys.H_total)

## Scrambling Map
m0_list = QDR.matrix_representation_ops(
    QDR.zero_charge_probabilities(sys.grids.total), sys.H_total)
m1_list = QDR.matrix_representation_ops(
    QDR.single_charge_probabilities(sys.grids.total), sys.H_total)
m2_list = QDR.matrix_representation_ops(
    QDR.double_charge_probabilities(sys.grids.total), sys.H_total)
t = [100, 200]
S0 = scrambling_map(sys, m0_list, ψ_res, hams.total, t)
S1 = scrambling_map(sys, m1_list, ψ_res, hams.total, t)
S2 = scrambling_map(sys, m2_list, ψ_res, hams.total, t)
S = vcat(S0, S1, S2)

## Is Σ00 singular vector?
v = Σ00 / norm(Σ00)
λ = dot(v, S'S * v) / dot(v, v)
residual = norm(S'S * v - λ * v) / norm(S'S * v)

## Overlap 
Pm = QDR.pauli_matrix(sys.Hs_main, sys.H_main)[1]
real.(svd(S).Vt*Pm / 2)
# The overlap between the first singular vector and Σ00 is close to 1

##Trace of effective measurements
S*Σ00 # Example: The trace of Pi2 is small when we have few electrons in the reservoir

## MSE terms
Σab = vec(Ps[:σz, :σz])
F = svd(S)
σE = 10 ^ -3
b = 0.0147
a = 0.0588
d = 4

T1 = Σab' * F.V * diagm((b * σE^2) ./ (b * F.S .^ 2 .+ σE ^ 2)) * F.V' * Σab
T2 = a * b * σE^4 * abs2(Σab' * F.V * diagm(1 ./ (b * F.S .^ 2 .+ σE ^ 2)) * F.V' * Σ00) /
     (1/d - a * σE^2 * Σ00' * F.V * diagm(1 ./ (b * F.S .^ 2 .+ σE ^ 2)) * F.V' * Σ00)

frac = T2/T1
frac_bound = a*d/b * (1 - abs2((F.Vt * Σ00)[1]) / d)

## W Terms

Σab = vec(Ps[:σz, :σz])
F = svd(S)
σE = 10 ^ -3
b = 0.0147
a = 0.0588
d = 4

A1 = F.U * diagm(b .* (F.S .^ 2) ./ (b .* F.S .^ 2 .+ σE^2)) * F.U'

A2_denom = 1 + a * Σ00' * F.V * diagm(F.S .^ 2 ./ (b .* F.S .^ 2 .+ σE^2)) * F.V' * Σ00
A2_nom = F.U * diagm(F.S .^ 3 ./ (b .* F.S .^ 2 .+ σE^2)) * F.V' * Σ00 * Σ00' * F.V *
         diagm(F.S ./ (b .* F.S .^ 2 .+ σE^2)) * F.U'
A2 = a*b/A2_denom .* A2_nom

Wnoisefree = Σab' * pinv(S)

norm(Wnoisefree - Wnoisefree*(A1 - A2)) / norm(Wnoisefree)
norm(Wnoisefree*A1 - Wnoisefree*(A1 - A2)) / norm(Wnoisefree)

