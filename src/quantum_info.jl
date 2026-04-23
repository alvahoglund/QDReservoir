
function mutual_information(ψ::AbstractVector, H, HA) #Assumes the other subsystem is the complement of HA
    ρmain = partial_trace(density_matrix(ψ), H => HA)
    2 * von_neumann(ρmain)
end
function mutual_information(ψ::AbstractVector, H, HA, HB)
    HC = FermionicHilbertSpaces.complementary_subsystem(H, tensor_product((HA, HB)))
    HC = subregion(HC, H)
    ρ = density_matrix(ψ)
    alg = FermionicHilbertSpaces.FullPartialTraceAlg() #use this for safety
    ρA = partial_trace(ρ, H => HA; alg)
    ρB = partial_trace(ρ, H => HB; alg)
    ρC = partial_trace(ρ, H => HC; alg)
    von_neumann(ρA) + von_neumann(ρB) - von_neumann(ρC)
end
function mutual_information(ρ_total::AbstractMatrix, H, HA, HB)
    HAB = tensor_product((HA, HB))
    HAB = subregion(HAB, H)
    alg = FermionicHilbertSpaces.FullPartialTraceAlg()
    ρAB = partial_trace(ρ_total, H => HAB; alg)
    ρ_main = partial_trace(ρAB, HAB => HA; complement = HB, alg)
    ρ_res = partial_trace(ρAB, HAB => HB; complement = HA, alg)
    S_main = von_neumann(ρ_main)
    S_res = von_neumann(ρ_res)
    S_total = von_neumann(ρAB)
    return S_main + S_res - S_total
end

von_neumann(ρ::LowRankMatrix; kwargs...) = von_neumann(Matrix(ρ), kwargs...)
function von_neumann(ρ::AbstractMatrix; tol = sqrt(eps(real(eltype(ρ)))))
    λ = eigvals(Hermitian(ρ))
    λ = λ[λ .> tol]
    S = -sum(λ .* log.(λ))
    return S
end
