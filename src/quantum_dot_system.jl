struct QuantumDotSystem
    coordinates_main
    coordinates_reservoir
    coordinates_total
    coordinates_intersection

    qn_main::Int
    qn_reservoir::Int
    qn_total::Int

    H_main_qn::FermionicHilbertSpaces.AbstractHilbertSpace
    H_main::FermionicHilbertSpaces.AbstractHilbertSpace
    H_main_a::FermionicHilbertSpaces.AbstractHilbertSpace
    H_main_b::FermionicHilbertSpaces.AbstractHilbertSpace

    H_reservoir::FermionicHilbertSpaces.AbstractHilbertSpace
    H_reservoir_qn::FermionicHilbertSpaces.AbstractHilbertSpace

    H_total::FermionicHilbertSpaces.AbstractHilbertSpace
    H_total_qn::FermionicHilbertSpaces.AbstractHilbertSpace

    f
end

labels(coordinates) = [(coordinate, spin) for coordinate in coordinates for spin in (:↑, :↓)]
sites(H) = unique(first.(keys(H)))

function tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    coordinates_main, coordinates_reservoir, coordinates_total, coordinates_intersection = generate_grid(nbr_dots_main, nbr_dots_res)
    qn_total = qn_reservoir + nbr_dots_main

    H_main_qn = tensor_product(hilbert_space(labels([coordinates_main[1]]), NumberConservation(1)), hilbert_space(labels([coordinates_main[2]]), NumberConservation(1)))
    H_main = hilbert_space(keys(H_main_qn), NumberConservation())
    H_main_a = hilbert_space(labels([coordinates_main[1]]), NumberConservation())
    H_main_b = hilbert_space(labels([coordinates_main[2]]), NumberConservation())
    H_reservoir = hilbert_space(labels(coordinates_reservoir), NumberConservation())
    H_total = hilbert_space(labels(coordinates_total), NumberConservation())
    H_total_qn = sector(qn_total, H_total)
    H_reservoir_qn = sector(qn_reservoir, H_reservoir)

    @fermions f

    QuantumDotSystem(coordinates_main, coordinates_reservoir, coordinates_total, coordinates_intersection,
        nbr_dots_main, qn_reservoir, qn_total,
        H_main_qn, H_main, H_main_a, H_main_b,
        H_reservoir,
        H_reservoir_qn,
        H_total, H_total_qn,
        f)
end

function generate_grid(nbr_dots_main::Int, nbr_dots_reservoir::Int)
    coordinates_main = [(1, i) for i in 1:nbr_dots_main]
    coordinates_reservoir = [
        (div(i - 1, nbr_dots_main) + 2, mod1(i, nbr_dots_main))
        for i in 1:nbr_dots_reservoir
    ]
    coordinates_total = vcat(coordinates_main, coordinates_reservoir)
    coordinates_intersection = vcat(coordinates_main, [coordinate for coordinate in coordinates_reservoir if coordinate[1] == 2])
    return coordinates_main, coordinates_reservoir, coordinates_total, coordinates_intersection
end