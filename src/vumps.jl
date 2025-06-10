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

function leftorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-14, maxiter = 100, kwargs...)
    χ, d, = size(A)
    Q, R = polar(reshape(C * reshape(A, χ, d * χ), χ * d, χ))
    AL = Array(reshape(Q, χ, d, χ))
    λ = norm(R)
    R ./= λ
    numiter = 1
    while norm(C .- R) > tol && numiter < maxiter
        ALbar = conj.(AL)
        _, vecs = eigsolve(R, 1, :LR; ishermitian = false, tol = tol, verbosity = 0, kwargs...) do X
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
        _, vecs = eigsolve(L, 1, :LR; ishermitian = false, tol = tol, verbosity = 0, kwargs...) do X
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

function AC2AL(x)
    χ, d, = size(x)
    d -= 1
    AC = x[:, 1 : end - 1, :]
    C = x[:, end, :]
    L1, = polar(reshape(AC, χ * d, χ))
    L2, = polar(C)
    reshape(L1 * L2', χ, d, χ)
end

function AC2AR(x)
    χ, d, = size(x)
    d -= 1
    AC = x[:, 1 : end - 1, :]
    C = x[:, end, :]
    _, R1 = polar(reshape(AC, χ, d * χ); rev = true)
    _, R2 = polar(C)
    reshape(R2' * R1, χ, d, χ)
end

struct UniformMPS <: Manifold end

function Optim.retract!(::UniformMPS, x)
    χ, d, = size(x)
    d -= 1
    AC = x[:, 1 : end - 1, :]
    C = x[:, end, :]
    invC = inv(C)
    AL = ein"ijk, kl -> ijl"(AC, invC)
    AR = ein"ij, jkl -> ikl"(invC, AC)
    AL, L, = leftorth(AL)
    R, AR, = rightorth(AR)
    AC .= ein"ij, jkl, lm -> ikm"(L, AC, R)
    C .= L * C * R
    AC ./= norm(AC)
    C ./= norm(C)
    x[:, 1 : end - 1, :] .= AC
    x[:, end, :] .= C
    x
end

function Optim.project_tangent!(::UniformMPS, dx, x)
    χ, d, = size(x)
    d -= 1
    AC = x[:, 1 : end - 1, :]
    C = x[:, end, :]
    U, S, V = svd(C)
    AC2 = ein"ij, jkl, lm -> ikm"(U', AC, V)
    U1, _, V1 = svd(reshape(AC2, χ * d, χ))
    U1 .= U1 * V1'
    U2, _, V2 = svd(reshape(AC2, χ, d * χ))
    V2 .= V2 * U2'
    dAC = ein"ij, jkl, lm -> ikm"(U', dx[:, 1 : end - 1, :], V)
    normdAC = norm(dAC)
    dC = U' * dx[:, end, :] * V
    S² = S .^ 2
    F = inv.(S² .+ S²')
    while true
        L1 = Diagonal(S) * U1' * reshape(dAC, χ * d, χ)
        L2 = Diagonal(S) * dC
        temp = (L1 .+ L1' .- (L2 .+ L2')) ./ 4
        if norm(temp) < 1e-14 * normdAC
            break
        end
        h = F .* temp
        dAC .-= reshape(U1 * Diagonal(S) * h, χ, d, χ)
        dC .+= Diagonal(S) * h

        R1 = reshape(dAC, χ, d * χ) * V2 * Diagonal(S)
        R2 = dC * Diagonal(S)
        h = F .* ((R1 .+ R1' .- (R2 .+ R2')) ./ 4)
        dAC .-= reshape(h * Diagonal(S) * V2', χ, d, χ)
        dC .+= h * Diagonal(S)
    end
    dAC .= ein"ij, jkl, lm -> ikm"(U, dAC, V')
    dC .= U * dC * V'
    dAC .-= AC .* real(dot(AC, dAC))
    dC .-= C .* real(dot(C, dC))
    dx[:, 1 : end - 1, :] .= dAC
    dx[:, end, :] .= dC
end

struct MixedCanonicalMPS{T}
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
    function fg!(F, G, x)
        val, (dx,) = withgradient(y -> local_energy(AC2AL(y), y[:, 1 : end - 1, :], h), x)
        if G !== nothing
            G .= dx
        end
        if F !== nothing
            return val
        end
    end
    res = optimize(Optim.only_fg!(fg!), cat(A.AC, reshape(A.C, χ, 1, χ); dims = 2), LBFGS(manifold = UniformMPS()), Optim.Options(g_abstol = tol, allow_f_increases = true, iterations = iterations))

    y = Optim.minimizer(res)
    AC = y[:, 1 : end - 1, :]
    C = y[:, end, :]
    AL = AC2AL(y)
    AR = AC2AR(y)
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