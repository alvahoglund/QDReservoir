const f = FermionicHilbertSpaces.SymbolicFermionBasis(:f, 0)
struct QuantumDotSystem
    grid::Any

    Hs_main::Any
    H_main::Any

    H_res::Any
    H_total::Any
end

function labels(coordinates)
    [(coordinate, spin) for coordinate in coordinates for spin in (:↑, :↓)]
end
sites(H) = unique(first.(keys(H)))

function tight_binding_system(nbr_dots_main, nbr_dots_res, qn_res)
    grid = generate_grid(nbr_dots_main, nbr_dots_res)
    qn_total = qn_res + nbr_dots_main

    Hs_main = [hilbert_space(labels((coordinate,)), NumberConservation(1))
               for coordinate in grid.main]
    H_main = tensor_product(Hs_main)
    H_res = hilbert_space(labels(grid.res), NumberConservation(qn_res))

    H_total = hilbert_space(labels(grid.total), NumberConservation(qn_total))

    QuantumDotSystem(grid,
        Hs_main, H_main, H_res, H_total)
end

function generate_grid(nbr_dots_main::Int, nbr_dots_res::Int)
    grid_main = [(1, i) for i in 1:nbr_dots_main]
    grid_res = [(div(i - 1, nbr_dots_main) + 2, mod1(i, nbr_dots_main))
                for i in 1:nbr_dots_res]
    grid_total = vcat(grid_main, grid_res)
    grid_intersection = vcat(grid_main,
        [coordinate for coordinate in grid_res if coordinate[1] == 2])
    return (; main = grid_main, res = grid_res,
        total = grid_total, intersection = grid_intersection)
end

qn_sector(H::SymmetricFockHilbertSpace) = H.symmetry.conserved_quantity.sectors[1]