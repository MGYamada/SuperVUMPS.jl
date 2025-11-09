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

function rightorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-12, maxiter = 100)
    χ, d, = size(A)
    Abar = conj(A)
    _, vecs1 = eigsolve(C * C', 1, :LR; ishermitian = false, tol = 1e-2tol, verbosity = 0) do X
        ein"ijk, (ljm, mk) -> li"(Abar, A, X)
    end
    ρ = vecs1[1]
    U, S, = svd(ρ)
    L = U * Diagonal(sqrt.(S)) * U'
    L ./= norm(L)
    f(X) = X .- polar(reshape(reshape(A, χ * d, χ) * X, χ, d * χ); rev = true)[1]
    numiter = 0
    while norm(f(L)) > tol && numiter < maxiter
        u, s, v = svd(reshape(reshape(A, χ * d, χ) * L, χ, d * χ))
        dL, = linsolve((x -> cat(real(x), imag(x); dims = 3))(f(L)); tol = 1e-2tol, verbosity = 0) do x
            (x -> cat(real(x), imag(x); dims = 3))(x[:, :, 1] .+ im .* x[:, :, 2] .- u * ((u' * ein"ijk, (ljm, mk) -> li"(conj(A), A, (x[:, :, 1] .+ im .* x[:, :, 2]) * L' .+ L * (x[:, :, 1] .+ im .* x[:, :, 2])') * u) ./ (s .+ s')) * u')
        end
        L .-= dL[:, :, 1] .+ im .* dL[:, :, 2]
        U, S, V = svd(L)
        L .= U * Diagonal(S) * U'
        L ./= norm(L)
        numiter += 1
    end
    _, R = polar(reshape(reshape(A, χ * d, χ) * L, χ, d * χ); rev = true)
    L, Array(reshape(R, χ, d, χ))
end

struct UniformMPS <: Manifold end

function Optim.retract!(::UniformMPS, AC; tol = 1e-12)
    χ, d, = size(AC)
    L, C = polar(reshape(AC, χ * d, χ))
    AL = reshape(L, χ, d, χ)
    C, = rightorth(AL, C; tol = tol)
    AC .= ein"ijk, kl -> ijl"(AL, C)
    AC ./= norm(AC)
end

function Optim.project_tangent!(::UniformMPS, dAC, AC; tol = 1e-12)
    χ, d, = size(AC)
    U1, S1, V1 = svd(reshape(AC, χ, d * χ))
    U2, S2, V2 = svd(reshape(AC, χ * d, χ) * U1)
    U2 .= U2 * V2'
    V2 .= U1
    sqrtS1 = sqrt.(S1)
    invsqrtS1 = inv.(sqrtS1)
    sqrtS2 = sqrt.(S2)
    invsqrtS2 = inv.(sqrtS2)
    K1 = Diagonal(invsqrtS1) * (U1' * reshape(dAC, χ, d * χ) * V1) * Diagonal(sqrtS1)
    K2 = Diagonal(invsqrtS2) * (V2' * reshape(dAC, χ * d, χ)' * U2) * Diagonal(sqrtS2)
    temp, = linsolve((x -> cat(real(x), imag(x); dims = 3))(K1 .+ K1' .- (K2 .+ K2')); ishermitian = true, isposdef = true, tol = tol, verbosity = 0) do x
        h = x[:, :, 1] .+ im .* x[:, :, 2]
        dac = reshape(U1 * (Diagonal(invsqrtS1) * (h .+ h') * Diagonal(sqrtS1)) * V1', χ, d, χ) .- reshape(U2 * (Diagonal(sqrtS2) * (h .+ h') * Diagonal(invsqrtS2)) * V2', χ, d, χ)
        k1 = Diagonal(invsqrtS1) * (U1' * reshape(dac, χ, d * χ) * V1) * Diagonal(sqrtS1)
        k2 = Diagonal(invsqrtS2) * (V2' * reshape(dac, χ * d, χ)' * U2) * Diagonal(sqrtS2)
        (x -> cat(real(x), imag(x); dims = 3))(k1 .+ k1' .- (k2 .+ k2'))
    end
    h = temp[:, :, 1] .+ im .* temp[:, :, 2]
    dAC .-= reshape(U1 * (Diagonal(invsqrtS1) * (h .+ h') * Diagonal(sqrtS1)) * V1', χ, d, χ) .- reshape(U2 * (Diagonal(sqrtS2) * (h .+ h') * Diagonal(invsqrtS2)) * V2', χ, d, χ)
    dAC .-= AC .* real(dot(AC, dAC))
end

struct MixedCanonicalMPS{T <: Complex}
    AL::Array{T, 3}
    AR::Array{T, 3}
    AC::Array{T, 3}
    C::Matrix{T}
end

function regularize_left(AL, ALbar, C, Cbar, h, χ; tol = 1e-12)
    r = ein"ij, kj -> ik"(Cbar, C)
    l = Matrix{ComplexF64}(I, χ, χ)

    initial = ein"ijk, (klm, (jlno, (inq, qor))) -> mr"(ALbar, ALbar, h, AL, AL)
    Lh, = linsolve(x -> x .- ein"(ij, ikl), jkm -> lm"(x, ALbar, AL) .+ ein"ij, ij -> "(x, r)[] .* l, initial .- ein"ij, ij -> "(initial, r)[] .* l; ishermitian = false, tol = tol, verbosity = 0)
    (Lh .+ Lh') ./ 2
end

function regularize_right(AR, ARbar, C, Cbar, h, χ; tol = 1e-12)
    l = ein"ij, ik -> jk"(Cbar, C)
    r = Matrix{ComplexF64}(I, χ, χ)

    initial = ein"ijk, (klm, (jlno, (pnq, qom))) -> ip"(ARbar, ARbar, h, AR, AR)
    Rh, = linsolve(x -> x .- ein"ijk, (mjl, kl) -> im"(ARbar, AR, x) .+ r .* ein"ij, ij -> "(l, x)[], initial .- r .* ein"ij, ij -> "(l, initial)[]; ishermitian = false, tol = tol, verbosity = 0)
    (Rh .+ Rh') ./ 2
end

function canonicalMPS(T, χ, d)
    U, _, V = svd(randn(T, χ * d, χ))
    AL = Array(reshape(U * V', χ, d, χ))
    C, AR = rightorth(AL)
    AC = ein"ijk, kl -> ijl"(AL, C)
    MixedCanonicalMPS(AL, AR, AC, C)
end

conjugateMPS(A) = MixedCanonicalMPS(conj(A.AL), conj(A.AR), conj(A.AC), conj(A.C))

local_energy(AL, AC, h::Array{T, 4}) where T = ein"ijk, (klm, (jlno, (inp, pom))) -> "(conj(AL), conj(AC), h, AL, AC)[]
local_energy(AL, AC, h::Array{T, 6}) where T = ein"ijk, (klm, (mno, (jlnpqr, (ips, (sqt, tro))))) -> "(conj(AL), conj(AL), conj(AC), h, AL, AL, AC)[]
local_energy(AL, AC, h::Array{T, 8}) where T = ein"ijk, (klm, (mno, (opq, (jlnprstu, (irv, (vsw, (wtx, xuq))))))) -> "(conj(AL), conj(AL), conj(AL), conj(AC), h, AL, AL, AL, AC)[]

function Hamiltonian_construction(h::Array{T, 4}, A, E; tol = 1e-12) where T
    χ, d, = size(A.AL)
    Abar = conjugateMPS(A)
    hr = h .- E .* ein"ij, kl -> ikjl"(Matrix{Float64}(I, d, d), Matrix{Float64}(I, d, d))
    Lh = regularize_left(A.AL, Abar.AL, A.C, Abar.C, hr, χ; tol = 1e-2tol)
    Rh = regularize_right(A.AR, Abar.AR, A.C, Abar.C, hr, χ; tol = 1e-2tol)
    HL = ein"ijk, (jlno, inp) -> klpo"(Abar.AL, hr, A.AL)
    HC = ein"ijk, (lmn, (jmop, (ioq, rpn))) -> klqr"(Abar.AL, Abar.AR, hr, A.AL, A.AR)
    HR = ein"klm, (jlno, pom) -> jknp"(Abar.AR, hr, A.AR)
    HAC = ein"klpo, ij -> klipoj"(HL, Matrix{Float64}(I, χ, χ)) .+ ein"jknp, hi -> hjkinp"(HR, Matrix{Float64}(I, χ, χ)) .+
    ein"ij, kl, mn -> ikmjln"(Lh, Matrix{Float64}(I, d, d), Matrix{Float64}(I, χ, χ)) .+ ein"ij, kl, mn -> ikmjln"(Matrix{Float64}(I, χ, χ), Matrix{Float64}(I, d, d), Rh)
    HC_rtn = HC .+ ein"ij, kl -> ikjl"(Lh, Matrix{Float64}(I, χ, χ)) .+ ein"ij, kl -> ikjl"(Matrix{Float64}(I, χ, χ), Rh)
    HAC, HC_rtn
end

function svumps(h::T, A; tol = 1e-8, iterations = 1000, Hamiltonian = false) where T
    χ, d, = size(A.AL)
    U, _, V = svd(A.C)
    AC = ein"ij, (jkl, lm) -> ikm"(U', A.AC, V)

    function fg!(F, G, x)
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
    res = optimize(Optim.only_fg!(fg!), AC, LBFGS(manifold = UniformMPS()), Optim.Options(g_abstol = tol, allow_f_increases = true, iterations = iterations))

    AC .= Optim.minimizer(res)
    L, C = polar(reshape(AC, χ * d, χ))
    AL = reshape(L, χ, d, χ)
    _, R = polar(reshape(AC, χ, d * χ); rev = true)
    AR = reshape(R, χ, d, χ)
    A = MixedCanonicalMPS(AL, AR, AC, C)

    E = real(local_energy(A.AL, A.AC, h))
    if Hamiltonian
        E, A, Hamiltonian_construction(h, A, E; tol = tol)...
    else
        E, A
    end
end

Zygote.@adjoint function svumps(h, A; kwargs...)
    X = svumps(h, A; kwargs...)
    _, A, = X
    X, function (Δ)
        if all(Δ[2 : end] .=== nothing)
            _, back = pullback(x -> real(local_energy(A.AL, A.AC, x)), h)
            (back(Δ[1])[1], nothing)
        else
            error("MPS/effective Hamiltonian differentiation not supported")
        end
    end
end