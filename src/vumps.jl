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

function rightorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-12, offset = 1e-8, maxiter = 100)
    χ, d, = size(A)
    Abar = conj(A)
    _, vecs1 = eigsolve(C * C', 1, :LR; ishermitian = false, tol = tol, krylovdim = 100, verbosity = 0) do X
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
            (x -> cat(real(x), imag(x); dims = 3))((1.0 + offset) .* (x[:, :, 1] .+ im .* x[:, :, 2]) .- u * ((u' * ein"ijk, (ljm, mk) -> li"(Abar, A, (x[:, :, 1] .+ im .* x[:, :, 2]) * L' .+ L * (x[:, :, 1] .+ im .* x[:, :, 2])') * u) ./ (s .+ s')) * u')
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

safesign(x::Number) = iszero(x) ? one(x) : sign(x)

struct UniformMPS <: Manifold end

function Optim.retract!(::UniformMPS, x; tol = 1e-12)
    χ, d, = size(x)
    d -= 1
    AC = x[:, 1 : end - 1, :]
    C = x[:, end, :]
    U, S, V = svd(reshape(AC, χ * d, χ))
    S ./= norm(S)
    f(X) = X .- svdvals(reshape(U * Diagonal(X) * V', χ, d * χ))
    while norm(f(S)) > tol
        S .-= jacobian(f, S)[1] \ f(S)
        @. S = abs(S)
        S ./= norm(S)
    end
    AC .= reshape(U * Diagonal(S) * V', χ, d, χ)
    U, = svd(reshape(AC, χ, d * χ))
    s = S .* safesign.(diag(U' * C * V))
    C .= U * Diagonal(s) * V'
    x[:, 1 : end - 1, :] .= AC
    x[:, end, :] .= C
    x
end

function Optim.project_tangent!(::UniformMPS, dx, x; tol = 1e-12)
    χ, d, = size(x)
    d -= 1
    AC = x[:, 1 : end - 1, :]
    dAC = dx[:, 1 : end - 1, :]
    C = x[:, end, :]
    dC = dx[:, end, :]
    H1 = ein"ijk, ljk -> li"(conj(dAC), AC) .+ ein"ijk, ljk -> li"(conj(AC), dAC) .- dC * C' .- C * dC'
    H2 = ein"ijk, ijl -> kl"(conj(dAC), AC) .+ ein"ijk, ijl -> kl"(conj(AC), dAC) .- dC' * C .- C' * dC
    temp, = linsolve(vcat(real(dot(AC, dAC)), real(dot(C, dC)), (x -> vcat(real(x), imag(x)))(vcat(vec(H1), vec(H2)))); ishermitian = true, isposdef = true, tol = tol, verbosity = 0) do x
        y = reshape(x[3 : end], :, 2)
        z = y[:, 1] .+ im .* y[:, 2]
        h = reshape(z, χ, χ, 2)
        dac = x[1] .* AC .+ ein"ijk, li -> ljk"(AC, h[:, :, 1] .+ h[:, :, 1]') .+ ein"ijk, kl -> ijl"(AC, h[:, :, 2] .+ h[:, :, 2]')
        dc = x[2] .* C .- ((h[:, :, 1] .+ h[:, :, 1]') * C .+ C * (h[:, :, 2] .+ h[:, :, 2]'))
        h1 = ein"ijk, ljk -> li"(conj(dac), AC) .+ ein"ijk, ljk -> li"(conj(AC), dac) .- dc * C' .- C * dc'
        h2 = ein"ijk, ijl -> kl"(conj(dac), AC) .+ ein"ijk, ijl -> kl"(conj(AC), dac) .- dc' * C .- C' * dc
        vcat(real(dot(AC, dac)), real(dot(C, dc)), (x -> vcat(real(x), imag(x)))(vcat(vec(h1), vec(h2))))
    end
    y = reshape(temp[3 : end], :, 2)
    z = y[:, 1] .+ im .* y[:, 2]
    h = reshape(z, χ, χ, 2)
    dAC .-= temp[1] .* AC .+ ein"ijk, li -> ljk"(AC, h[:, :, 1] .+ h[:, :, 1]') .+ ein"ijk, kl -> ijl"(AC, h[:, :, 2] .+ h[:, :, 2]')
    dC .-= temp[2] .* C .- ((h[:, :, 1] .+ h[:, :, 1]') * C .+ C * (h[:, :, 2] .+ h[:, :, 2]'))
    dx[:, 1 : end - 1, :] .= dAC
    dx[:, end, :] .= dC
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
    U, S, V = svd(A.C)
    AC = ein"ij, (jkl, lm) -> ikm"(U', A.AC, V)

    function fg!(F, G, x)
        val, (dx,) = withgradient(x) do y
            ac = y[:, 1 : end - 1, :]
            c = y[:, end, :]
            lac, = polar(reshape(ac, χ * d, χ))
            lc, = polar(c)
            al = reshape(lac * lc', χ, d, χ)
            real(local_energy(al, ac, h))
        end
        if G !== nothing
            G .= dx
        end
        if F !== nothing
            return val
        end
    end
    res = optimize(Optim.only_fg!(fg!), cat(AC, reshape(Diagonal(S), χ, 1, χ); dims = 2), LBFGS(manifold = UniformMPS()), Optim.Options(g_abstol = tol, allow_f_increases = true, iterations = iterations))

    x = Optim.minimizer(res)
    AC .= x[:, 1 : end - 1, :]
    C = x[:, end, :]
    LAC, = polar(reshape(AC, χ * d, χ))
    LC, = polar(C)
    AL = reshape(LAC * LC', χ, d, χ)
    _, RAC = polar(reshape(AC, χ, d * χ); rev = true)
    _, RC = polar(C; rev = true)
    AR = reshape(RC' * RAC, χ, d, χ)
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