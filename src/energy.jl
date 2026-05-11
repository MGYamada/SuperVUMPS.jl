local_energy_two_body(AL, AC, h) = ein"ijk, (klm, (jlno, (inp, pom))) -> "(conj(AL), conj(AC), h, AL, AC)[]
local_energy_three_body(AL, AC, h) = ein"ijk, (klm, (mno, (jlnpqr, (ips, (sqt, tro))))) -> "(conj(AL), conj(AL), conj(AC), h, AL, AL, AC)[]
local_energy_four_body(AL, AC, h) = ein"ijk, (klm, (mno, (opq, (jlnprstu, (irv, (vsw, (wtx, xuq))))))) -> "(conj(AL), conj(AL), conj(AL), conj(AC), h, AL, AL, AL, AC)[]

local_energy(AL, AC, h::AbstractArray{T, 4}) where T = local_energy_two_body(AL, AC, h)
local_energy(AL, AC, h::AbstractArray{T, 6}) where T = local_energy_three_body(AL, AC, h)
local_energy(AL, AC, h::AbstractArray{T, 8}) where T = local_energy_four_body(AL, AC, h)
