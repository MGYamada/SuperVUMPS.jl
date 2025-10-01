function polar(A; rev = false)
    U, S, V = _svd(A)
    if rev
        U * Diagonal(S) * U', U * V'
    else
        U * V', V * Diagonal(S) * V'
    end
end

_svd(A) = svd(A) # to avoid piracy
Zygote.@adjoint _svd(A) = svd_back(A)

function svd_back(A; η = 1e-40)
    U, S, V = svd(A)
    (U, S, V), function (Δ)
        ΔA = Δ[2] === nothing ? zeros(eltype(A), size(A)...) : U * Diagonal(Δ[2]) * V'
        if Δ[1] !== nothing || Δ[3] !== nothing
            S² = S .^ 2
            invS = @. S / (S² + η)
            F = S²' .- S²
            @. F /= (F ^ 2 + η)
            if Δ[1] !== nothing
                J = F .* (U' * Δ[1])
                ΔA .+= U * (J .+ J') * Diagonal(S) * V'
                ΔA .+= (I - U * U') * Δ[1] * Diagonal(invS) * V'
            end
            if Δ[3] !== nothing
                K = F .* (V' * Δ[3])
                ΔA .+= U * Diagonal(S) * (K .+ K') * V'
                L = Diagonal(diag(V' * Δ[3]))
                ΔA .+= 0.5 .* U * Diagonal(invS) * (L' .- L) * V'
                ΔA .+= U * Diagonal(invS) * Δ[3]' * (I - V * V')
            end
        end
        (ΔA,)
    end
end

ϕ(x) = iszero(x) ? one(x) : x / abs(x)

function Sinkhorn(A; tol = 1e-14)
    n = size(A, 1)
    L = ones(eltype(A), n)
    R = ones(eltype(A), n)
    sumold = sum(A)
    while true
        temp = map(x -> ϕ(x)', vec(sum(A; dims = 2)))
        L = L .* temp
        A = Diagonal(temp) * A
        temp = map(x -> ϕ(x)', vec(sum(A; dims = 1)))
        R = R .* temp
        A = A * Diagonal(temp)
        sumnew = sum(A)
        if abs(sumnew - sumold) < tol
            break
        end
        sumold = sumnew
    end
    L, R
end

function Sinkhorn!(A)
    n = size(A, 1)
    F = [exp(2π * im / n * (i - 1) * (j - 1)) / sqrt(n) for i in 1 : n, j in 1 : n]
    U = F' * A * F
    U[1, :] .= 0
    U[:, 1] .= 0
    U[1, 1] = 1
    u, = polar(U[2 : end, 2 : end])
    U[2 : end, 2 : end] .= u
    Anew = F * U * F'
    L = Anew * A'
    A .= Anew
    L
end

function leftorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-14, maxiter = 100, kwargs...)
    χ, d, = size(A)
    Q, R = polar(reshape(C * reshape(A, χ, d * χ), χ * d, χ))
    AL = Array(reshape(Q, χ, d, χ))
    λ = norm(R)
    R ./= λ
    numiter = 1
    while norm(C .- R) > tol && numiter < maxiter
        ALbar = conj.(AL)
        _, vecs = eigsolve(R, 1, :LR; ishermitian = false, tol = tol, maxiter = 1, verbosity = 0, kwargs...) do X
            ein"(ij, ikl), jkm -> lm"(X, ALbar, A)
        end
        C = vecs[1]
        _, C = polar(C)
        Q, R = polar(reshape(C * reshape(A, χ, d * χ), χ * d, χ))
        AL = reshape(Q, χ, d, χ)
        λ = norm(R)
        R ./= λ
        numiter += 1
    end
    if eltype(A) <: Real
        real.(AL), real.(R), λ
    else
        AL, R, λ
    end
end

function rightorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-14, maxiter = 100, kwargs...)
    χ, d, = size(A)
    L, Q = polar(reshape(reshape(A, χ * d, χ) * C, χ, d * χ); rev = true)
    AR = Array(reshape(Q, χ, d, χ))
    λ = norm(L)
    L ./= λ
    numiter = 1
    while norm(C .- L) > tol && numiter < maxiter
        ARbar = conj.(AR)
        _, vecs = eigsolve(L, 1, :LR; ishermitian = false, tol = tol, maxiter = 1, verbosity = 0, kwargs...) do X
            ein"mkj, (lki, ji) -> ml"(A, ARbar, X)
        end
        C = vecs[1]
        C, = polar(C; rev = true)
        L, Q = polar(reshape(reshape(A, χ * d, χ) * C, χ, d * χ); rev = true)
        AR = reshape(Q, χ, d, χ)
        λ = norm(L)
        L ./= λ
        numiter += 1
    end
    if eltype(A) <: Real
        real.(L), real.(AR), λ
    else
        L, AR, λ
    end
end

struct UniformMPS <: Manifold end

function Optim.retract!(::UniformMPS, AC)
    χ, d, = size(AC)
    ac, C = ACproj(AC) # fix later
    AC .= ac
    invC = inv(C)
    AL = ein"ijk, kl -> ijl"(AC, invC)
    AR = ein"ij, jkl -> ikl"(invC, AC)
    AL, L, = leftorth(AL)
    R, AR, = rightorth(AR)
    C .= L * C * R
    C ./= norm(C)
    AC .= ein"ijk, kl -> ijl"(AL, C)
    U, _, V = svdfix(C; fix = :U)
    L1 = Sinkhorn!(U)
    L2 = Sinkhorn!(V)
    AC .= ein"ij, (jkl, lm) -> ikm"(L1, AC, L2')
end

function Optim.project_tangent!(::UniformMPS, dAC, AC)
    χ, d, = size(AC)
    U1, _, V1 = svd(reshape(AC, χ, d * χ))
    U2, _, V2 = svd(reshape(AC, χ * d, χ))
    M = Matrix{eltype(AC)}(I, χ, χ)
    for i in 1 : χ, j in 1 : χ
        M[i, j] -= (transpose(U1[:, j]) * reshape(U2'[i, :], χ, d)) * (reshape(V1'[j, :], d, χ) * V2[:, i])
    end
    M .+= M'
    K1 = U1' * reshape(dAC, χ, d * χ) * V1
    K2 = U2' * reshape(dAC, χ * d, χ) * V2
    U, S, V = svd(M)
    temp = V[:, 1 : end - 1] * (Diagonal(inv.(S[1 : end - 1])) * (U'[1 : end - 1, :] * diag(K1 .- K2)))
    dAC .-= reshape(U1 * Diagonal(temp) * V1', χ, d, χ) .- reshape(U2 * Diagonal(temp) * V2', χ, d, χ)
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

function svdfix(A; fix = :U)
    U, S, V = _svd(A)
    if fix == :U
        phase = map(x -> ϕ(x)', vec(sum(U; dims = 1)))
        U = U * Diagonal(phase)
        V = V * Diagonal(phase)
    elseif fix == :V
        phase = map(x -> ϕ(x)', vec(sum(V; dims = 1)))
        U = U * Diagonal(phase)
        V = V * Diagonal(phase)
    end
    U, S, V
end

function ACproj(AC)
    χ, d, = size(AC)
    U, S1, = svdfix(reshape(AC, χ, d * χ); fix = :U)
    _, S2, V = svdfix(reshape(AC, χ * d, χ); fix = :V)
    S = (S1 .+ S2) ./ 2
    L1, = Sinkhorn(U)
    L2, = Sinkhorn(V)
    ein"ij, (jkl, lm) -> ikm"(Diagonal(L1), AC, Diagonal(L2)'), Diagonal(L1) * U * Diagonal(S) * V' * Diagonal(L2)' # fix later
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
    U, _, V = svdfix(A.C; fix = :U)
    AC = ein"ij, (jkl, lm) -> ikm"(U', A.AC, V)

    function fg!(F, G, x)
        val, (dx,) = withgradient(x) do y
            ac, c = ACproj(y)
            L1, = polar(reshape(ac, χ * d, χ)) # fix leter
            L2, = polar(c) # fix later
            real(local_energy(reshape(L1 * L2', χ, d, χ), ac, h))
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
    L, = polar(reshape(AC, χ * d, χ))
    C, R = polar(reshape(AC, χ, d * χ); rev = true)
    AL = reshape(L, χ, d, χ)
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