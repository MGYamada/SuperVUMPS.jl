safesign(x::Number) = iszero(x) ? one(x) : sign(x)
copyltu(A) = tril(A, -1)' .+ tril(A, -1) .+ Diagonal(real(diag(A)))

function qrpos(A)
    Q, R = _qr(A)
    phases = safesign.(diag(R))
    Q * Diagonal(phases), Diagonal(phases)' * R
end

function _qr(A) # to avoid piracy
    Q, R = qr(A)
    Matrix(Q), R
end

Zygote.@adjoint function _qr(A)
    Q, R = _qr(A)
    (Q, R), function (Δ)
        m, n = size(A)
        @assert m >= n
        ΔQ, ΔR = Δ
        if ΔR === nothing
            M = -ΔQ' * Q
            ΔA = (ΔQ .+ Q * copyltu(M)) / R'
        elseif ΔQ === nothing
            M = R * ΔR'
            ΔA = (Q * copyltu(M)) / R'
        else
            M = R * ΔR' .- ΔQ' * Q
            ΔA = (ΔQ .+ Q * copyltu(M)) / R'
        end
        (ΔA,)
    end
end

function lqpos(A)
    L, Q = lq(A)
    phases = safesign.(diag(L))
    L * Diagonal(phases)', Diagonal(phases) * Q
end

function leftorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-14, kwargs...)
    χ, d, = size(A)
    Q, R = qrpos(reshape(C * reshape(A, χ, d * χ), χ * d, χ))
    AL = Array(reshape(Q, χ, d, χ))
    λ = norm(R)
    R ./= λ
    δ = norm(C .- R)
    while δ > tol
        ALbar = conj.(AL)
        _, vecs = eigsolve(R, 1, :LR; ishermitian = false, tol = 1e-2δ, verbosity = 0, kwargs...) do X
            ein"(ij, ikl), jkm -> lm"(X, ALbar, A)
        end
        C = vecs[1]
        _, C = qrpos(C)
        Q, R = qrpos(reshape(C * reshape(A, χ, d * χ), χ * d, χ))
        AL = reshape(Q, χ, d, χ)
        λ = norm(R)
        R ./= λ
        δ = norm(C .- R)
    end
    if eltype(A) <: Real
        real(AL), real(R), λ
    else
        AL, R, λ
    end
end

function rightorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-14, kwargs...)
    χ, d, = size(A)
    L, Q = lqpos(reshape(reshape(A, χ * d, χ) * C, χ, d * χ))
    AR = Array(reshape(Q, χ, d, χ))
    λ = norm(L)
    L ./= λ
    δ = norm(C .- L)
    while δ > tol
        ARbar = conj.(AR)
        _, vecs = eigsolve(L, 1, :LR; ishermitian = false, tol = 1e-2δ, verbosity = 0, kwargs...) do X
            ein"mkj, (lki, ji) -> ml"(A, ARbar, X)
        end
        C = vecs[1]
        C, = lqpos(C)
        L, Q = lqpos(reshape(reshape(A, χ * d, χ) * C, χ, d * χ))
        AR = reshape(Q, χ, d, χ)
        λ = norm(L)
        L ./= λ
        δ = norm(C .- L)
    end
    if eltype(A) <: Real
        real(L), real(AR), λ
    else
        L, AR, λ
    end
end

struct UniformMPS <: Manifold end

function Optim.retract!(::UniformMPS, x; tol = 1e-12)
    χ, d, = size(x)
    d -= 1
    AC = x[:, 1 : d, :]
    C = x[:, end, :]
    L, Q = lqpos(C)
    LAC, = qrpos(reshape(AC, χ * d, χ) * Q')
    LC, = qrpos(C * Q')
    AL = reshape(LAC * LC', χ, d, χ)
    C, = rightorth(AL, L; tol = tol)
    C .= C * Q
    AC .= ein"ijk, kl -> ijl"(AL, C)
    x[:, 1 : d, :] .= AC
    x[:, end, :] .= C
    x
end

function Optim.project_tangent!(::UniformMPS, dx, x; tol = 1e-12)
    χ, d, = size(x)
    d -= 1
    AC = x[:, 1 : d, :]
    C = x[:, end, :]
    dAC = dx[:, 1 : d, :]
    dC = dx[:, end, :]
    dC1 = dC * C' .+ C * dC' .- (reshape(dAC, χ, d * χ) * reshape(AC, χ, d * χ)' .+ reshape(AC, χ, d * χ) * reshape(dAC, χ, d * χ)')
    dC2 = dC' * C .+ C' * dC .- (reshape(dAC, χ * d, χ)' * reshape(AC, χ * d, χ) .+ reshape(AC, χ * d, χ)' * reshape(dAC, χ * d, χ))
    temp, = linsolve((x -> cat(real(x), imag(x); dims = 4))(cat(dC1, dC2; dims = 3)); ishermitian = true, isposdef = true, tol = tol, verbosity = 0) do x
        h1 = x[:, :, 1, 1] .+ im .* x[:, :, 1, 2]
        h2 = x[:, :, 2, 1] .+ im .* x[:, :, 2, 2]
        dc = (h1 .+ h1') * C .+ C * (h2 .+ h2')
        dac = -(ein"ij, jkl -> ikl"(h1 .+ h1', AC) .+ ein"ijk, kl -> ijl"(AC, h2 .+ h2'))
        dc1 = dc * C' .+ C * dc' .- (reshape(dac, χ, d * χ) * reshape(AC, χ, d * χ)' .+ reshape(AC, χ, d * χ) * reshape(dac, χ, d * χ)')
        dc2 = dc' * C .+ C' * dc .- (reshape(dac, χ * d, χ)' * reshape(AC, χ * d, χ) .+ reshape(AC, χ * d, χ)' * reshape(dac, χ * d, χ))
        (x -> cat(real(x), imag(x); dims = 4))(cat(dc1, dc2; dims = 3))
    end
    h1 = temp[:, :, 1, 1] .+ im .* temp[:, :, 1, 2]
    h2 = temp[:, :, 2, 1] .+ im .* temp[:, :, 2, 2]
    dC .-= (h1 .+ h1') * C .+ C * (h2 .+ h2')
    dAC .+= ein"ij, jkl -> ikl"(h1 .+ h1', AC) .+ ein"ijk, kl -> ijl"(AC, h2 .+ h2')
    dC .-= C .* real(dot(C, dC))
    dAC .-= AC .* real(dot(AC, dAC))
    dx[:, 1 : d, :] .= dAC
    dx[:, end, :] .= dC
    dx
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
    A = randn(T, χ, d, χ)
    AL, = leftorth(A)
    C, AR, = rightorth(AL)
    AC = ein"ijk, kl -> ijl"(AL, C)
    MixedCanonicalMPS(AL, AR, AC, C)
end

conjugateMPS(A) = MixedCanonicalMPS(conj.(A.AL), conj.(A.AR), conj.(A.AC), conj.(A.C))

local_energy(AL, AC, h::Array{T, 4}) where T = ein"ijk, (klm, (jlno, (inp, pom))) -> "(conj.(AL), conj.(AC), h, AL, AC)[]
local_energy(AL, AC, h::Array{T, 6}) where T = ein"ijk, (klm, (mno, (jlnpqr, (ips, (sqt, tro))))) -> "(conj.(AL), conj.(AL), conj.(AC), h, AL, AL, AC)[]
local_energy(AL, AC, h::Array{T, 8}) where T = ein"ijk, (klm, (mno, (opq, (jlnprstu, (irv, (vsw, (wtx, xuq))))))) -> "(conj.(AL), conj.(AL), conj.(AL), conj.(AC), h, AL, AL, AL, AC)[]

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
    function fg!(F, G, x)
        val, (dx,) = withgradient(x) do y
            ac = y[:, 1 : d, :]
            c = y[:, end, :]
            lac, = qrpos(reshape(ac, χ * d, χ))
            lc, = qrpos(c)
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
    res = optimize(Optim.only_fg!(fg!), cat(A.AC, reshape(A.C, χ, 1, χ); dims = 2), LBFGS(manifold = UniformMPS()), Optim.Options(g_abstol = tol, allow_f_increases = true, iterations = iterations))

    x = Optim.minimizer(res)
    AC = x[:, 1 : d, :]
    C = x[:, end, :]
    LAC, = qrpos(reshape(AC, χ * d, χ))
    LC, = qrpos(C)
    AL = reshape(LAC * LC', χ, d, χ)
    _, RAC = lqpos(reshape(AC, χ, χ * d))
    _, RC = lqpos(C)
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