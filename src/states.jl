## =================== Vaccum state ================ 
function vac_state(H)
    v0 = spzeros(dim(H))
    v0[FermionicHilbertSpaces.state_index(FockNumber(0), H)] = 1.0
    return v0
end

## =================== 2 dot main system states =============
singlet(f) = 1 / √2 * (f[(1, 1), :↑]' * f[(1, 2), :↓]' - f[(1, 1), :↓]' * f[(1, 2), :↑]')
triplet_0(f) = 1 / √2 * (f[(1, 1), :↑]' * f[(1, 2), :↓]' + f[(1, 1), :↓]' * f[(1, 2), :↑]')
triplet_plus(f) = 1 / √2 * ((f[(1, 1), :↑]' * f[(1, 2), :↑]' + f[(1, 1), :↓]' * f[(1, 2), :↓]'))
triplet_minus(f) = 1 / √2 * ((f[(1, 1), :↑]' * f[(1, 2), :↑]' - f[(1, 1), :↓]' * f[(1, 2), :↓]'))

function def_state(state_name, H, f)
    vac_ind = FermionicHilbertSpaces.state_index(FockNumber(0), H)
    H2, v0 = if ismissing(vac_ind)
        Haux = hilbert_space(keys(H), push!(copy(basisstates(H)), FockNumber(UInt(0))))
        Haux, vac_state(Haux)
    else
        H, vac_state(H)
    end
    v = matrix_representation(state_name(f), H2; projection=true) * v0
    if ismissing(vac_ind)
        v = v[1:end-1]
    end
    ρ = v * v'
    ρ = ρ / norm(ρ)
    return ρ
end


function max_mixed_state(H, f)
    v0 = vac_state(H)
    states = [matrix_representation(f[(1, 1), σ1]'f[(1, 2), σ2]', H) * v0 for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓]]
    ρ_mixed = 1 / 2 * sum(state * state' for state in states)
    return ρ_mixed
end

werner_state(state_name, p, H, f) = (1 - p) * def_state(state_name, H, f) + p * max_mixed_state(H, f)

random_state(H) = normalize!(randn(ComplexF64, dim(H)))
random_product_state(Hs, H) = generalized_kron(random_state.(Hs), Hs => H)
random_product_state(sys::QuantumDotSystem) = random_product_state(sys.Hs_main, sys.H_main)
function random_separable_state(nbr_states, Hs, H)
    p = rand(nbr_states)
    p = p ./ sum(p)
    ρ_sep = sum(p[i] * density_matrix(random_product_state(Hs, H)) for i ∈ 1:nbr_states)
    return ρ_sep
end
random_separable_state(N, sys::QuantumDotSystem) = random_separable_state(N, sys.Hs_main, sys.H_main)
density_matrix(v::AbstractVector) = v * v'
density_matrix(ρ::AbstractMatrix) = ρ

function hilbert_schmidt_ensemble(H)
    d = dim(H)
    X = randn(d, d)
    ρ = X'X / tr(X'X)
    return ρ
end

## =============== Ground States ====================
abstract type DiagonalizationAlg end
struct ExactDiagonalizationAlg <: DiagonalizationAlg end

function eig_state(m::AbstractMatrix, n, ::ExactDiagonalizationAlg)
    eigenvalues, eigenvectors = eigen(Matrix(m))
    eigenvectors[:, n]
end

using ArnoldiMethod
struct ArnoldiAlg <: DiagonalizationAlg end
function eig_state(m::AbstractMatrix, n, ::ArnoldiAlg; kwargs...)
    decomp, history = try
        partialschur(Hermitian(m), nev=n, which=:SR; kwargs...)
    catch e
        @warn e "Trying to increase mindim and restarts"
        println(m)
        partialschur(Hermitian(m), nev=n, which=:SR; kwargs..., mindim=40, maxdim=size(m, 1), restarts=1000)
    end
    # @show history
    vals, vecs = partialeigen(decomp)
    idx = sortperm(real(vals))[n]
    return vecs[:, idx]
    # vals, vecs
end

ground_state(m, alg=ArnoldiAlg()) = eig_state(m, 1, alg)
eig_state(m, n) = eig_state(m, n, ArnoldiAlg())
