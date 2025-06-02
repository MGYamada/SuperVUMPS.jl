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

function leftorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-14, maxiter = 100, kwargs...) # fix later
    D, d, = size(A)
    Q, R = polar(reshape(C * reshape(A, D, d * D), D * d, D))
    AL = reshape(Q, D, d, D)
    λ = norm(R)
    R /= λ
    numiter = 1
    while norm(C .- R) > tol && numiter < maxiter
        vals2, vecs2 = eigsolve(R, 1, :LR; ishermitian = false, tol = tol, maxiter = 1, kwargs...) do X
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

function Optim.project_tangent!(::UniformMPS, dAC, AC; η = 1e-40)
    χ, d, = size(AC)
    U1, S1, V1 = svd(reshape(AC, χ, d * χ))
    U2, S2, V2 = svd(reshape(AC, χ * d, χ) * U1)
    U2 .= U2 * V2'
    V2 .= U1
    sqrtS1 = sqrt.(S1)
    invsqrtS1 = inv.(sqrtS1)
    sqrtS2 = sqrt.(S2)
    invsqrtS2 = inv.(sqrtS2)

    K1 = Diagonal(invsqrtS1) * (U1' * reshape(dAC, χ, d * χ) * V1) * Diagonal(sqrtS1) # numerical stabilization
    K2 = Diagonal(invsqrtS2) * (V2' * reshape(dAC, χ * d, χ)' * U2) * Diagonal(sqrtS2) # numerical stabilization
    temp, = linsolve(K1 .+ K1' .- (K2 .+ K2'); ishermitian = true, isposdef = true, tol = 1e-14) do x
        dac = reshape(U1 * Diagonal(invsqrtS1) * (x + x') * Diagonal(sqrtS1) * V1', χ, d, χ) .- reshape(U2 * Diagonal(sqrtS2) * (x + x') * Diagonal(invsqrtS2) * V2', χ, d, χ)
        K1 = Diagonal(invsqrtS1) * (U1' * reshape(dac, χ, d * χ) * V1) * Diagonal(sqrtS1) # numerical stabilization
        K2 = Diagonal(invsqrtS2) * (V2' * reshape(dac, χ * d, χ)' * U2) * Diagonal(sqrtS2) # numerical stabilization
        K1 .+ K1' .- (K2 .+ K2')
    end
    dAC .-= reshape(U1 * Diagonal(invsqrtS1) * (temp + temp') * Diagonal(sqrtS1) * V1', χ, d, χ) .- reshape(U2 * Diagonal(sqrtS2) * (temp + temp') * Diagonal(invsqrtS2) * V2', χ, d, χ)
    dAC .-= AC .* real(dot(AC, dAC))
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
    Lh, = linsolve(x -> x .- ein"(ij, ikl), jkm -> lm"(x, ALbar, AL) .+ ein"ij, ij -> "(x, r)[] .* l, initial .- ein"ij, ij -> "(initial, r)[] .* l; ishermitian = false, tol = tol)
    (Lh .+ Lh') ./ 2
end

function regularize_right(AR, ARbar, C, Cbar, h, χ; tol = 1e-12)
    l = ein"ij, ik -> jk"(Cbar, C)
    r = Matrix{ComplexF64}(I, χ, χ)

    initial = ein"ijk, (klm, (jlno, (pnq, qom))) -> ip"(ARbar, ARbar, h, AR, AR)
    Rh, = linsolve(x -> x .- ein"ijk, (mjl, kl) -> im"(ARbar, AR, x) .+ r .* ein"ij, ij -> "(l, x)[], initial .- r .* ein"ij, ij -> "(l, initial)[]; ishermitian = false, tol = tol)
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

function svumps(h::T, A; tol = 1e-8, Niter = 1000, Hamiltonian = false) where T
    χ, d, = size(A.AL)
    Abar = conjugateMPS(A)
    U, P = polar(A.C)
    AC = ein"ij, jkl -> ikl"(P, A.AR) # polar gauge
    retractAC!(AC, χ, d)

    function fg!(F, G, x)
        val, (dx,) = withgradient(y -> (local_energy(reshape(polar(reshape(y, χ * d, χ))[1], χ, d, χ), y, h)), x)
        if G !== nothing
            G .= dx
        end
        if F !== nothing
            return val
        end
    end
    res = optimize(Optim.only_fg!(fg!), AC, LBFGS(manifold = UniformMPS()), Optim.Options(g_tol = tol, allow_f_increases = true, iterations = Niter))

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