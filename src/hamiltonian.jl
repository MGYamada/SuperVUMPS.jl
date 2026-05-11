identity_matrix(T, n) = Matrix{T}(I, n, n)

function regularize_left(AL, ALbar, C, Cbar, h, χ; tol = 1e-12)
    T = promote_type(eltype(AL), eltype(C), eltype(h))
    r = ein"ij, kj -> ik"(Cbar, C)
    l = identity_matrix(T, χ)

    initial = ein"ijk, (klm, (jlno, (inq, qor))) -> mr"(ALbar, ALbar, h, AL, AL)
    Lh, = linsolve(x -> x .- ein"(ij, ikl), jkm -> lm"(x, ALbar, AL) .+ ein"ij, ij -> "(x, r)[] .* l, initial .- ein"ij, ij -> "(initial, r)[] .* l; ishermitian = false, tol = tol, verbosity = 0)
    (Lh .+ Lh') ./ 2
end

function regularize_right(AR, ARbar, C, Cbar, h, χ; tol = 1e-12)
    T = promote_type(eltype(AR), eltype(C), eltype(h))
    l = ein"ij, ik -> jk"(Cbar, C)
    r = identity_matrix(T, χ)

    initial = ein"ijk, (klm, (jlno, (pnq, qom))) -> ip"(ARbar, ARbar, h, AR, AR)
    Rh, = linsolve(x -> x .- ein"ijk, (mjl, kl) -> im"(ARbar, AR, x) .+ r .* ein"ij, ij -> "(l, x)[], initial .- r .* ein"ij, ij -> "(l, initial)[]; ishermitian = false, tol = tol, verbosity = 0)
    (Rh .+ Rh') ./ 2
end

function construct_hamiltonian(h::AbstractArray{T, 4}, A, E; tol = 1e-12) where T
    χ, d, = size(A.AL)
    Abar = conjugateMPS(A)
    matrix_type = promote_type(eltype(h), typeof(E))
    virtual_type = promote_type(eltype(A.AL), eltype(h), typeof(E))
    Id = identity_matrix(matrix_type, d)
    Iχ = identity_matrix(virtual_type, χ)

    hr = h .- E .* ein"ij, kl -> ikjl"(Id, Id)
    Lh = regularize_left(A.AL, Abar.AL, A.C, Abar.C, hr, χ; tol = 1e-2tol)
    Rh = regularize_right(A.AR, Abar.AR, A.C, Abar.C, hr, χ; tol = 1e-2tol)
    HL = ein"ijk, (jlno, inp) -> klpo"(Abar.AL, hr, A.AL)
    HC = ein"ijk, (lmn, (jmop, (ioq, rpn))) -> klqr"(Abar.AL, Abar.AR, hr, A.AL, A.AR)
    HR = ein"klm, (jlno, pom) -> jknp"(Abar.AR, hr, A.AR)
    HAC = ein"klpo, ij -> klipoj"(HL, Iχ) .+ ein"jknp, hi -> hjkinp"(HR, Iχ) .+
        ein"ij, kl, mn -> ikmjln"(Lh, Id, Iχ) .+ ein"ij, kl, mn -> ikmjln"(Iχ, Id, Rh)
    HC_rtn = HC .+ ein"ij, kl -> ikjl"(Lh, Iχ) .+ ein"ij, kl -> ikjl"(Iχ, Rh)
    HAC, HC_rtn
end

Hamiltonian_construction(h, A, E; kwargs...) = construct_hamiltonian(h, A, E; kwargs...)
