function polar(A; rev = false)
    U, S, V = svd(A)
    if rev
        U * Diagonal(S) * U', U * V'
    else
        U * V', V * Diagonal(S) * V'
    end
end

Zygote.@adjoint function polar(A; rev = false)
    U, S, V = svd(A)
    if rev
        d = size(A, 1)
        P, Q = U * Diagonal(S) * U', U * V'
        (P, Q), function (Δ)
            if Δ[2] !== nothing
                invP = U * Diagonal(inv.(S)) * U'
                ΔB = Δ[2] * A'
                ΔA = invP' * Δ[2]
                ΔP = -invP' * ΔB * invP'
            else
                ΔA = zeros(eltype(A), size(A)...)
                ΔP = zeros(eltype(P), size(P)...)
            end
            if Δ[1] !== nothing
                ΔP += Δ[1]
            end
            ΔAA = U * ((U' * ΔP * U) ./ (S .+ S')) * U'
            ΔA += (ΔAA .+ ΔAA') * A
            (ΔA,)
        end
    else
        d = size(A, 2)
        Q, P = U * V', V * Diagonal(S) * V'
        (Q, P), function (Δ)
            if Δ[1] !== nothing
                invP = V * Diagonal(inv.(S)) * V'
                ΔA = Δ[1] * invP'
                ΔB = A' * Δ[1]
                ΔP = -invP' * ΔB * invP'
            else
                ΔA = zeros(eltype(A), size(A)...)
                ΔP = zeros(eltype(P), size(P)...)
            end
            if Δ[2] !== nothing
                ΔP += Δ[2]
            end
            ΔAA = V * ((V' * ΔP * V) ./ (S .+ S')) * V'
            ΔA += A * (ΔAA .+ ΔAA')
            (ΔA,)
        end
    end
end

_verbosity(verbose) = verbose isa Bool ? (verbose ? 1 : 0) : verbose
_show_trace(verbose) = verbose isa Bool ? verbose : verbose > 0

function _hermitian_difference(A, B)
    C = A .+ A'
    C .-= B
    C .-= B'
    C
end

function _project_tangent_basis(AC)
    χ, d, = size(AC)
    U1, S1, V1 = svd(reshape(AC, χ, d * χ))
    U2, S2, V2 = svd(reshape(AC, χ * d, χ) * U1)
    U2 .= U2 * V2'
    V2 .= U1
    U1, S1, V1, U2, S2, V2
end

function _scaled_tangent_components(dAC, U1, V1, U2, V2, DinvsqrtS1, DsqrtS1, DinvsqrtS2, DsqrtS2)
    χ, d, = size(dAC)
    K1 = DinvsqrtS1 * (U1' * reshape(dAC, χ, d * χ) * V1) * DsqrtS1
    K2 = DinvsqrtS2 * (V2' * reshape(dAC, χ * d, χ)' * U2) * DsqrtS2
    K1, K2
end

function _project_tangent_correction(h, χ, d, U1, V1, U2, V2, DinvsqrtS1, DsqrtS1, DinvsqrtS2, DsqrtS2)
    hp = h .+ h'
    reshape(U1 * (DinvsqrtS1 * hp * DsqrtS1) * V1', χ, d, χ) .- reshape(U2 * (DsqrtS2 * hp * DinvsqrtS2) * V2', χ, d, χ)
end

function rightorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-12, maxiter = 100, verbose = false, kwargs...)
    χ, d, = size(A)
    Abar = conj(A)
    verbosity = _verbosity(verbose)
    _, vecs1 = eigsolve(C * C', 1, :LR; ishermitian = false, tol = 1e-2tol, verbosity, kwargs...) do X
        ein"ijk, (ljm, mk) -> li"(Abar, A, X)
    end
    ρ = vecs1[1]
    U, S, = svd(ρ)
    C = U * Diagonal(sqrt.(S)) * U'
    C ./= norm(C)
    L, R = polar(reshape(reshape(A, χ * d, χ) * C, χ, d * χ); rev = true)
    AR = Array(reshape(R, χ, d, χ))
    δ = norm(C .- L)
    numiter = 0
    while δ > tol && numiter < maxiter
        ARbar = conj(AR)
        _, vecs = eigsolve(L, 1, :LR; ishermitian = false, tol = 1e-2δ, verbosity, kwargs...) do X
            ein"ijk, (ljm, mk) -> li"(ARbar, A, X)
        end
        C = vecs[1]
        C, = polar(C; rev = true)
        L, R = polar(reshape(reshape(A, χ * d, χ) * C, χ, d * χ); rev = true)
        AR .= reshape(R, χ, d, χ)
        L ./= norm(L)
        δ = norm(C .- L)
        numiter += 1
    end
    L, AR
end

struct UniformMPS <: Manifold
    verbosity::Int
end

UniformMPS(; verbose = false) = UniformMPS(_verbosity(verbose))

function Optim.retract!(m::UniformMPS, AC; tol = 1e-12)
    χ, d, = size(AC)
    L, C = polar(reshape(AC, χ * d, χ))
    AL = reshape(L, χ, d, χ)
    ALbar = conj(AL)
    _, vecs1 = eigsolve(C * C', 1, :LR; ishermitian = false, tol = 1e-2tol, verbosity = m.verbosity) do x
        ein"ijk, (ljm, mk) -> li"(ALbar, AL, x)
    end
    X = vecs1[1]
    U, S, = svd(X)
    C .= U * Diagonal(sqrt.(S)) * U'
    C ./= norm(C)
    AC .= ein"ijk, kl -> ijl"(AL, C)
    AC ./= norm(AC)
end

function Optim.project_tangent!(m::UniformMPS, dAC, AC; tol = 1e-12)
    χ, d, = size(AC)
    U1, S1, V1, U2, S2, V2 = _project_tangent_basis(AC)
    sqrtS1 = sqrt.(S1)
    invsqrtS1 = inv.(sqrtS1)
    sqrtS2 = sqrt.(S2)
    invsqrtS2 = inv.(sqrtS2)
    DsqrtS1 = Diagonal(sqrtS1)
    DinvsqrtS1 = Diagonal(invsqrtS1)
    DsqrtS2 = Diagonal(sqrtS2)
    DinvsqrtS2 = Diagonal(invsqrtS2)
    K1, K2 = _scaled_tangent_components(dAC, U1, V1, U2, V2, DinvsqrtS1, DsqrtS1, DinvsqrtS2, DsqrtS2)
    temp, = linsolve(_hermitian_difference(K1, K2); ishermitian = true, isposdef = true, tol = tol, verbosity = m.verbosity) do h
        dac = _project_tangent_correction(h, χ, d, U1, V1, U2, V2, DinvsqrtS1, DsqrtS1, DinvsqrtS2, DsqrtS2)
        k1, k2 = _scaled_tangent_components(dac, U1, V1, U2, V2, DinvsqrtS1, DsqrtS1, DinvsqrtS2, DsqrtS2)
        _hermitian_difference(k1, k2)
    end
    dAC .-= _project_tangent_correction(temp, χ, d, U1, V1, U2, V2, DinvsqrtS1, DsqrtS1, DinvsqrtS2, DsqrtS2)
    dAC .-= AC .* real(dot(AC, dAC))
end

struct MixedCanonicalMPS{T <: Complex}
    AL::Array{T, 3}
    AR::Array{T, 3}
    AC::Array{T, 3}
    C::Matrix{T}
end

function regularize_left(AL, ALbar, C, Cbar, h, χ; tol = 1e-12, verbose = false)
    r = ein"ij, kj -> ik"(Cbar, C)
    l = Matrix{eltype(C)}(I, χ, χ)

    initial = ein"ijk, (klm, (jlno, (inq, qor))) -> mr"(ALbar, ALbar, h, AL, AL)
    Lh, = linsolve(x -> x .- ein"(ij, ikl), jkm -> lm"(x, ALbar, AL) .+ ein"ij, ij -> "(x, r)[] .* l, initial .- ein"ij, ij -> "(initial, r)[] .* l; ishermitian = false, tol = tol, verbosity = _verbosity(verbose))
    (Lh .+ Lh') ./ 2
end

function regularize_right(AR, ARbar, C, Cbar, h, χ; tol = 1e-12, verbose = false)
    l = ein"ij, ik -> jk"(Cbar, C)
    r = Matrix{eltype(C)}(I, χ, χ)

    initial = ein"ijk, (klm, (jlno, (pnq, qom))) -> ip"(ARbar, ARbar, h, AR, AR)
    Rh, = linsolve(x -> x .- ein"ijk, (mjl, kl) -> im"(ARbar, AR, x) .+ r .* ein"ij, ij -> "(l, x)[], initial .- r .* ein"ij, ij -> "(l, initial)[]; ishermitian = false, tol = tol, verbosity = _verbosity(verbose))
    (Rh .+ Rh') ./ 2
end

function canonicalMPS(T, χ, d; verbose = false)
    U, _, V = svd(randn(T, χ * d, χ))
    AL = Array(reshape(U * V', χ, d, χ))
    C, AR = rightorth(AL; verbose)
    AC = ein"ijk, kl -> ijl"(AL, C)
    MixedCanonicalMPS(AL, AR, AC, C)
end

conjugateMPS(A) = MixedCanonicalMPS(conj(A.AL), conj(A.AR), conj(A.AC), conj(A.C))

_energy_eltype(AL, AC, h) = promote_type(eltype(AL), eltype(AC), eltype(h))
_energy_scalar(x, ::Type{T}) where T = x[]::T

local_energy(AL, AC, h::Array{T, 4}) where T = _energy_scalar(ein"ijk, (klm, (jlno, (inp, pom))) -> "(conj(AL), conj(AC), h, AL, AC), _energy_eltype(AL, AC, h))
local_energy(AL, AC, h::Array{T, 6}) where T = _energy_scalar(ein"ijk, (klm, (mno, (jlnpqr, (ips, (sqt, tro))))) -> "(conj(AL), conj(AL), conj(AC), h, AL, AL, AC), _energy_eltype(AL, AC, h))
local_energy(AL, AC, h::Array{T, 8}) where T = _energy_scalar(ein"ijk, (klm, (mno, (opq, (jlnprstu, (irv, (vsw, (wtx, xuq))))))) -> "(conj(AL), conj(AL), conj(AL), conj(AC), h, AL, AL, AL, AC), _energy_eltype(AL, AC, h))

function _shift_hamiltonian(h, E, d)
    Id = Matrix{Float64}(I, d, d)
    h .- E .* ein"ij, kl -> ikjl"(Id, Id), Id
end

function _regularized_environments(A, Abar, hr, χ; tol, verbose)
    Lh = regularize_left(A.AL, Abar.AL, A.C, Abar.C, hr, χ; tol = 1e-2tol, verbose)
    Rh = regularize_right(A.AR, Abar.AR, A.C, Abar.C, hr, χ; tol = 1e-2tol, verbose)
    Lh, Rh
end

function _effective_hamiltonian_terms(A, Abar, hr)
    HL = ein"ijk, (jlno, inp) -> klpo"(Abar.AL, hr, A.AL)
    HC = ein"ijk, (lmn, (jmop, (ioq, rpn))) -> klqr"(Abar.AL, Abar.AR, hr, A.AL, A.AR)
    HR = ein"klm, (jlno, pom) -> jknp"(Abar.AR, hr, A.AR)
    HL, HC, HR
end

function _assemble_effective_hamiltonians(HL, HC, HR, Lh, Rh, Id, Iχ)
    HAC = ein"klpo, ij -> klipoj"(HL, Iχ) .+ ein"jknp, hi -> hjkinp"(HR, Iχ) .+
    ein"ij, kl, mn -> ikmjln"(Lh, Id, Iχ) .+ ein"ij, kl, mn -> ikmjln"(Iχ, Id, Rh)
    HC_rtn = HC .+ ein"ij, kl -> ikjl"(Lh, Iχ) .+ ein"ij, kl -> ikjl"(Iχ, Rh)
    HAC, HC_rtn
end

function Hamiltonian_construction(h::Array{T, 4}, A, E; tol = 1e-12, verbose = false) where T
    χ, d, = size(A.AL)
    Abar = conjugateMPS(A)
    Iχ = Matrix{Float64}(I, χ, χ)
    hr, Id = _shift_hamiltonian(h, E, d)
    Lh, Rh = _regularized_environments(A, Abar, hr, χ; tol, verbose)
    HL, HC, HR = _effective_hamiltonian_terms(A, Abar, hr)
    HAC, HC_rtn = _assemble_effective_hamiltonians(HL, HC, HR, Lh, Rh, Id, Iχ)
    TH = promote_type(eltype(A.AL), eltype(h))
    HAC::Array{TH, 6}, HC_rtn::Array{TH, 4}
end

function _initial_ac(A)
    U, _, V = svd(A.C)
    ein"ij, (jkl, lm) -> ikm"(U', A.AC, V)
end

function _objective_gradient!(F, G, x, h, χ, d)
    val, (dx,) = withgradient(x) do ac
        l, = polar(reshape(ac, χ * d, χ))
        al = reshape(l, χ, d, χ)
        real(local_energy(al, ac, h))
    end
    if G !== nothing
        G .= dx
    end
    if F !== nothing
        return val
    end
end

function _mixed_canonical_from_ac(AC)
    χ, d, = size(AC)
    L, = polar(reshape(AC, χ * d, χ))
    AL = reshape(L, χ, d, χ)
    C, R = polar(reshape(AC, χ, d * χ); rev = true)
    AR = reshape(R, χ, d, χ)
    MixedCanonicalMPS(AL, AR, AC, C)
end

function _optimize_mps(h, A; tol, iterations, verbose)
    χ, d, = size(A.AL)
    AC = _initial_ac(A)
    fg! = (F, G, x) -> _objective_gradient!(F, G, x, h, χ, d)
    options = Optim.Options(g_abstol = tol, allow_f_increases = true, iterations = iterations, show_trace = _show_trace(verbose))
    res = optimize(only_fg!(fg!), AC, LBFGS(manifold = UniformMPS(; verbose)), options)
    AC .= Optim.minimizer(res)
    _mixed_canonical_from_ac(AC)
end

function svumps(h::T, A; tol = 1e-8, iterations = 1000, verbose = false) where T
    # Zygote.ignore is required for the Hellmann-Feynman gradient: gradients
    # should flow through the final expectation value, not through the optimizer.
    A0 = A
    Aopt = ignore() do
        _optimize_mps(h, A0; tol, iterations, verbose)
    end
    Aout = Aopt::typeof(A)

    E = real(local_energy(Aout.AL, Aout.AC, h))
    E, Aout
end

function svumps_hamiltonian(h::T, A; tol = 1e-8, iterations = 1000, verbose = false) where T
    E, A = svumps(h, A; tol, iterations, verbose)
    E, A, Hamiltonian_construction(h, A, E; tol = tol, verbose)...
end
