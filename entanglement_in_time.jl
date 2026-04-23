# # Entanglent in time
using FermionicHilbertSpaces, LinearAlgebra
N = 3
@fermions f
H = hilbert_space(f, 1:N, ParityConservation())
function kitaev_chain(f, N, μ, t, Δ, U)
    sum(t * f[i]' * f[i + 1] + hc for i in 1:(N - 1)) +
    sum(Δ * f[i] * f[i + 1] + hc for i in 1:(N - 1)) +
    sum(U * f[i]' * f[i] * f[i + 1]' * f[i + 1] for i in 1:(N - 1)) +
    sum(μ[i] * f[i]' * f[i] for i in 1:N)
end

U = 4.0
t = 1.0
δΔ = 0.4
Δ = t + U / 2 - δΔ # slightly detuned from the sweet spot
μ = fill(-U / 2, N) # edge chemical potential
μ[2:(N - 1)] .= -U # bulk chemical potential
hsym = kitaev_chain(f, N, μ, t, Δ, U)
ham = matrix_representation(hsym, H)
vals, vecs = eigen(Matrix(ham))
##
A = 1:2
HA = hilbert_space(f, A)
B = 3:N
HB = hilbert_space(f, B)
using TensorOperations
function calc_TAB(H, HA, HB, U, rho)
    UAB = reshape(U, H => (HA, HB))
    rhoAB = reshape(rho, H => (HA, HB))
    @tensor TAB[a, b, a', b'] := rhoAB[a, 1, 2, 3] * UAB[4, b', 2, 3] *
                                 conj(UAB[4, b, a', 1])
    reshape(TAB, (HA, HB), H)
end

##
tB = 0.5
U = vecs' * exp(-1im * Diagonal(vals) * tB) * vecs
ψ = vecs[:, 1]
ρ = ψ * ψ'
TAB = calc_TAB(H, HA, HB, U, ρ)
tr(TAB) ≈ 1# Should have trace 1
partial_trace(TAB, H => HA) ≈ partial_trace(ρ, H => HA) # Partial trace should match partial trace of rho

##
ts = range(0, 2, 100)
data = stack(ts) do t
    U = vecs' * exp(-1im * Diagonal(vals) * t) * vecs
    TAB = calc_TAB(H, HA, HB, U, ρ)
    norm(TAB^2), norm(TAB - TAB')
end
##
using Plots
plot(ts, data', ylims = (0, 1))