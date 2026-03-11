struct QuantumDotSystem
    grid::Any

    Hs_main::Any
    H_main::Any

    H_reservoir::Any
    H_total::Any

    f::Any
end

function labels(coordinates)
    [(coordinate, spin) for coordinate in coordinates for spin in (:↑, :↓)]
end
sites(H) = unique(first.(keys(H)))

function tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    grid = generate_grid(nbr_dots_main, nbr_dots_res)
    qn_total = qn_reservoir + nbr_dots_main

    Hs_main = [hilbert_space(labels((coordinate,)), NumberConservation(1))
               for coordinate in grid.main]
    H_main = tensor_product(Hs_main)
    H_reservoir = hilbert_space(labels(grid.res), NumberConservation(qn_reservoir))

    H_total = hilbert_space(labels(grid.total), NumberConservation(qn_total))

    @fermions f

    QuantumDotSystem(grid,
        Hs_main, H_main, H_reservoir, H_total,
        f)
end

function generate_grid(nbr_dots_main::Int, nbr_dots_reservoir::Int)
    coordinates_main = [(1, i) for i in 1:nbr_dots_main]
    coordinates_reservoir = [(div(i - 1, nbr_dots_main) + 2, mod1(i, nbr_dots_main))
                             for i in 1:nbr_dots_reservoir]
    coordinates_total = vcat(coordinates_main, coordinates_reservoir)
    coordinates_intersection = vcat(coordinates_main,
        [coordinate for coordinate in coordinates_reservoir if coordinate[1] == 2])
    return (; main = coordinates_main, res = coordinates_reservoir,
        total = coordinates_total, intersection = coordinates_intersection)
end

qn_sector(H::SymmetricFockHilbertSpace) = H.symmetry.conserved_quantity.sectors[1]