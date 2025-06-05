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
            ΔAA = sylvester(P, P, -ΔP)
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
            ΔAA = sylvester(P, P, -ΔP)
            ΔA += A * (ΔAA .+ ΔAA')
            (ΔA,)
        end
    end
end

function leftorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-14, maxiter = 100, kwargs...) # fix later
    D, d, = size(A)
    Q, R = polar(reshape(C * reshape(A, D, d * D), D * d, D))
    AL = reshape(Q, D, d, D)
    λ = norm(R)
    R /= λ
    numiter = 1
    while norm(C .- R) > tol && numiter < maxiter
        vals2, vecs2 = eigsolve(R, 1, :LR; ishermitian = false, tol = tol, maxiter = 1, verbosity = 0, kwargs...) do X
            ein"(ij, ikl), jkm -> lm"(X, conj.(AL), A)
        end
        C = vecs2[1]
        _, C = polar(C)
        Q, R = polar(reshape(C * reshape(A, D, d * D), D * d, D))
        AL = reshape(Q, D, d, D)
        λ = norm(R)
        R /= λ
        numiter += 1
    end
    C = R
    if eltype(A) <: Real
        real.(AL), real.(C), λ
    else
        AL, C, λ
    end
end

function rightorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-14, kwargs...) # fix later
    AL, C, λ = leftorth(ein"ijk -> kji"(A), transpose(C); tol = tol, kwargs...)
    Array(transpose(C)), ein"ijk -> kji"(AL), λ
end

function gauge_fixing(AC)
    χ, d, = size(AC)
    U, = svd(reshape(AC, χ, d * χ))
    _, _, V = svd(reshape(AC, χ * d, χ))
    phase = map(x -> x / abs(x), diag(V' * U))
    U, V * Diagonal(phase)
end

Zygote.@adjoint LinearAlgebra.svd(A) = svd_back(A)

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

function retractAC!(AC, χ, d) # polar gauge
    AC1 = reshape(AC, χ * d, χ)
    AC2 = Array(reshape(AC, χ, d * χ)')
    U, V, Q, D1, D2, R0 = svd(AC1, AC2)
    X = (R0 * Q') ./ sqrt(2)
    W, C = polar(X)
    AL = reshape((U * D1 * W) .* sqrt(2), χ, d, χ)
    AR = reshape((V * D2 * W)' .* sqrt(2), χ, d, χ)
    AL, L, = leftorth(AL)
    AC .= ein"ij, jkl -> ikl"(L, AC)
    C .= L * C
    R, AR, = rightorth(AR)
    AC .= ein"ijk, kl -> ijl"(AC, R)
    C .= C * R
    U, P = polar(C)
    AC .= ein"ij, jkl -> ikl"(P, AR)
    AC ./= norm(AC)
end

struct UniformMPS <: Manifold
end

function Optim.retract!(::UniformMPS, AC)
    χ, d, = size(AC)
    U, V = gauge_fixing(AC)
    W = U * V'
    AC .= ein"ijk, kl -> ijl"(AC, W')
    retractAC!(AC, χ, d)
    AC .= ein"ijk, kl -> ijl"(AC, W)
end

function Optim.project_tangent!(::UniformMPS, dAC, AC)
    χ, d, = size(AC)
    U1, _, V1 = svd(reshape(AC, χ, d * χ))
    U2, _, V2 = svd(reshape(AC, χ * d, χ))
    K1 = U1' * reshape(dAC, χ, d * χ) * V1
    K2 = U2' * reshape(dAC, χ * d, χ) * V2
    temp, = linsolve(diag(K1 .- K2); ishermitian = true, isposdef = true, tol = 1e-14, verbosity = 0) do x
        dac = reshape(U1 * Diagonal(x) * V1', χ, d, χ) .- reshape(U2 * Diagonal(x) * V2', χ, d, χ)
        K1 = U1' * reshape(dac, χ, d * χ) * V1
        K2 = U2' * reshape(dac, χ * d, χ) * V2
        diag(K1 .- K2) .+ 1e-14 .* x
    end
    dAC .-= reshape(U1 * Diagonal(temp) * V1', χ, d, χ) .- reshape(U2 * Diagonal(temp) * V2', χ, d, χ)
    dAC .-= AC .* real(dot(AC, dAC))
end

function AC2AL(AC)
    χ, d, = size(AC)
    U, V = gauge_fixing(AC)
    AC2 = ein"ijk, kl -> ijl"(AC, V * U')
    L, = polar(reshape(AC2, χ * d, χ))
    reshape(L, χ, d, χ)
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
    AC = randn(T, χ, d, χ)
    retractAC!(AC, χ, d)
    L, = polar(reshape(AC, χ * d, χ))
    C, R = polar(reshape(AC, χ, d * χ); rev = true)
    AL = reshape(L, χ, d, χ)
    AR = reshape(R, χ, d, χ)
    MixedCanonicalMPS(AL, AR, AC, C)
end

conjugateMPS(A) = MixedCanonicalMPS(conj.(A.AL), conj.(A.AR), conj.(A.AC), conj.(A.C))

local_energy(AL, AC, h::Array{T, 4}) where T = real(ein"ijk, (klm, (jlno, (inp, pom))) -> "(conj.(AL), conj.(AC), h, AL, AC)[])
local_energy(AL, AC, h::Array{T, 6}) where T = real(ein"ijk, (klm, (mno, (jlnpqr, (ips, (sqt, tro))))) -> "(conj.(AL), conj.(AL), conj.(AC), h, AL, AL, AC)[])
local_energy(AL, AC, h::Array{T, 8}) where T = real(ein"ijk, (klm, (mno, (opq, (jlnprstu, (irv, (vsw, (wtx, xuq))))))) -> "(conj.(AL), conj.(AL), conj.(AL), conj.(AC), h, AL, AL, AL, AC)[])

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
    Abar = conjugateMPS(A)
    U, P = polar(A.C)
    AC = ein"ij, jkl -> ikl"(P, A.AR) # polar gauge
    retractAC!(AC, χ, d)

    function fg!(F, G, x)
        val, (dx,) = withgradient(y -> (local_energy(AC2AL(y), y, h)), x)
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

    E = local_energy(A.AL, A.AC, h)
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
            _, back = pullback(x -> local_energy(A.AL, A.AC, x), h)
            (back(Δ[1])[1], nothing)
        else
            error("MPS/effective Hamiltonian differentiation not supported")
        end
    end
end