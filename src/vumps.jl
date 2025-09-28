function polar(A; rev = false)
    U, S, V = svd(A)
    if rev
        U * Diagonal(S) * U', U * V'
    else
        U * V', V * Diagonal(S) * V'
    end
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

function retractAC!(AC, χ, d)
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

struct UniformMPS <: Manifold end

function Optim.retract!(::UniformMPS, AC)
    χ, d, = size(AC)
    retractAC!(AC, χ, d)
end

function Optim.project_tangent!(::UniformMPS, dAC, AC)
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
    temp, = linsolve(K1 .+ K1' .- (K2 .+ K2'); ishermitian = true, isposdef = true, tol = 1e-14) do x
        dac = reshape(U1 * (Diagonal(invsqrtS1) * (x + x') * Diagonal(sqrtS1)) * V1', χ, d, χ) .- reshape(U2 * (Diagonal(sqrtS2) * (x + x') * Diagonal(invsqrtS2)) * V2', χ, d, χ)
        K1 = Diagonal(invsqrtS1) * (U1' * reshape(dac, χ, d * χ) * V1) * Diagonal(sqrtS1)
        K2 = Diagonal(invsqrtS2) * (V2' * reshape(dac, χ * d, χ)' * U2) * Diagonal(sqrtS2)
        K1 .+ K1' .- (K2 .+ K2')
    end
    dAC .-= reshape(U1 * (Diagonal(invsqrtS1) * (temp + temp') * Diagonal(sqrtS1)) * V1', χ, d, χ) .- reshape(U2 * (Diagonal(sqrtS2) * (temp + temp') * Diagonal(invsqrtS2)) * V2', χ, d, χ)
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
    A = randn(T, χ, d, χ)
    AL, = leftorth(A)
    C, AR, = rightorth(AL)
    AC = ein"ijk, kl -> ijl"(AL, C)
    MixedCanonicalMPS(AL, AR, AC, C)
end

conjugateMPS(A) = MixedCanonicalMPS(conj.(A.AL), conj.(A.AR), conj.(A.AC), conj.(A.C))

function fixedpointlinearbackward(next, g, args...)
    _, back = pullback(next, g, args...)
    back1 = x -> back(x)[1]
    back2 = x -> back(x)[2 : end]
    function (Δ)
        grad, = linsolve(Δ; ishermitian = false, verbosity = 0) do x
            x .- back1(x)
        end
        back2(grad)
    end
end

function fixedpointlinear(f, g, args...; kwargs...)
    λs, Gs, = eigsolve(x -> f(x, args...), g, 1, :LR; ishermitian = false, verbosity = 0, kwargs...)
    Gs[1]
end

@Zygote.adjoint function fixedpointlinear(f, g, args...; kwargs...)
    G = fixedpointlinear(f, g, args...; kwargs...)
    G, Δ -> (nothing, nothing, fixedpointlinearbackward(f, G, args...)(Δ)...)
end

function AL2R(AL, R)
    χ, = size(AL)
    R = fixedpointlinear(R, AL) do r, al
        ein"ijk, (ljm, km) -> il"(conj.(al), al, r)
    end
    R
end

local_energy(AL, R::Matrix{T}, h::Array{T, 4}) where T = ein"ijk, (klm, (jlno, (inp, (poq, mq)))) -> "(conj.(AL), conj.(AL), h, AL, AL, R)[]
local_energy(AL, R::Matrix{T}, h::Array{T, 6}) where T = ein"ijk, (klm, (mno, (jlnpqr, (ips, (sqt, (tru, ou)))))) -> "(conj.(AL), conj.(AL), conj.(AL), h, AL, AL, AL, R)[]
local_energy(AL, R::Matrix{T}, h::Array{T, 8}) where T = ein"ijk, (klm, (mno, (opq, (jlnprstu, (irv, (vsw, (wtx, (xuy, qy)))))))) -> "(conj.(AL), conj.(AL), conj.(AL), conj.(AL), h, AL, AL, AL, AL, R)[]

local_energy(AL, AC::Array{T, 3}, h::Array{T, 4}) where T = ein"ijk, (klm, (jlno, (inp, pom))) -> "(conj.(AL), conj.(AC), h, AL, AC)[]
local_energy(AL, AC::Array{T, 3}, h::Array{T, 6}) where T = ein"ijk, (klm, (mno, (jlnpqr, (ips, (sqt, tro))))) -> "(conj.(AL), conj.(AL), conj.(AC), h, AL, AL, AC)[]
local_energy(AL, AC::Array{T, 3}, h::Array{T, 8}) where T = ein"ijk, (klm, (mno, (opq, (jlnprstu, (irv, (vsw, (wtx, xuq))))))) -> "(conj.(AL), conj.(AL), conj.(AL), conj.(AC), h, AL, AL, AL, AC)[]

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

function svumps(h::T, A; tol = 1e-8, iterations = 100, Hamiltonian = false, hybrid = true) where T
    χ, d, = size(A.AL)
    R = Matrix{eltype(A.AL)}(I, χ, χ)
    function fg!(F, G, x)
        val, (dx,) = withgradient(x) do y
            R = AL2R(reshape(y, χ, d, χ), R)
            real(local_energy(reshape(y, χ, d, χ), R, h) / tr(R))
        end
        if G !== nothing
            G .= dx
        end
        if F !== nothing
            return val
        end
    end
    res = optimize(Optim.only_fg!(fg!), reshape(A.AL, χ * d, χ), LBFGS(manifold = Stiefel(), linesearch = BackTracking()), Optim.Options(g_abstol = tol, allow_f_increases = true, iterations = iterations))

    AL = Array(reshape(Optim.minimizer(res), χ, d, χ))
    C, AR, = rightorth(AL)
    AC = ein"ijk, kl -> ijl"(AL, C)
    A = MixedCanonicalMPS(AL, AR, AC, C)

    if hybrid
        U, P = polar(A.C)
        AC = ein"ij, jkl -> ikl"(P, A.AR) # polar gauge
        retractAC!(AC, χ, d)

        function fg2!(F, G, x)
            val, (dx,) = withgradient(y -> real(local_energy(reshape(polar(reshape(y, χ * d, χ))[1], χ, d, χ), y, h)), x)
            if G !== nothing
                G .= dx
            end
            if F !== nothing
                return val
            end
        end
        res = optimize(Optim.only_fg!(fg2!), AC, LBFGS(manifold = UniformMPS()), Optim.Options(g_tol = tol, allow_f_increases = true, iterations = iterations))
    end

    R .= AL2R(A.AL, R)
    E = real(local_energy(A.AL, R, h) / tr(R))
    if Hamiltonian
        E, A, R, Hamiltonian_construction(h, A, E; tol = tol)...
    else
        E, A, R
    end
end

Zygote.@adjoint function svumps(h, A; kwargs...)
    X = svumps(h, A; kwargs...)
    _, A, R, = X
    X, function (Δ)
        if all(Δ[2 : end] .=== nothing)
            _, back = pullback(x -> real(local_energy(A.AL, R, x) / tr(R)), h)
            (back(Δ[1])[1], nothing)
        else
            error("MPS/effective Hamiltonian differentiation not supported")
        end
    end
end