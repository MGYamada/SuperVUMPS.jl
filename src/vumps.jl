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

# function Sinkhorn!(A; tol = 1e-14)
#     n = size(A, 1)
#     L = ones(eltype(A), n)
#     R = ones(eltype(A), n)
#     sumold = sum(A)
#     while true
#         temp = map(x -> ϕ(x)', vec(sum(A; dims = 2)))
#         L .*= temp
#         A .= Diagonal(temp) * A
#         temp = map(x -> ϕ(x)', vec(sum(A; dims = 1)))
#         R .*= temp
#         A .= A * Diagonal(temp)
#         sumnew = sum(A)
#         if abs(sumnew - sumold) < tol
#             break
#         end
#         sumold = sumnew
#     end
#     L, R
# end

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

function gauge_fixing(AC)
    χ, d, = size(AC)
    U, S1, = svdfix(reshape(AC, χ, d * χ); fix = :U)
    _, S2, V = svdfix(reshape(AC, χ * d, χ); fix = :V)
    S = (S1 .+ S2) ./ 2
    U * Diagonal(S) * V'
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

struct UniformMPS <: Manifold end

function Optim.retract!(::UniformMPS, AC)
    χ, d, = size(AC)
    C = gauge_fixing(AC)
    invC = inv(C) # fix later
    AL = ein"ijk, kl -> ijl"(AC, invC)
    AR = ein"ij, jkl -> ikl"(invC, AC)
    AL, L, = leftorth(AL)
    R, AR, = rightorth(AR)
    C = L * C * R
    C ./= norm(C)
    AC .= ein"ijk, kl -> ijl"(AL, C)
    U, _, V = svdfix(C; fix = :U)
    L1 = Sinkhorn!(U)
    L2 = Sinkhorn!(V)
    AC .= ein"(ij, jkl), lm -> ikm"(L1, AC, L2')
end

function Optim.project_tangent!(::UniformMPS, dAC, AC; η = 1e-40)
    # O(χ⁴) algorithm by M. G. Yamada
    χ, d, = size(AC)
    Optim.retract!(UniformMPS(), AC)
    U1, S1, V1 = svdfix(reshape(AC, χ, d * χ); fix = :U)
    U2, S2, V2 = svdfix(reshape(AC, χ * d, χ); fix = :V)
    S = (S1 .+ S2) ./ 2
    S² = S .^ 2
    F = S²' .- S²
    @. F /= (F ^ 2 + η)
    dU1 = U1 * (F .* (x -> x .+ x')(U1' * reshape(dAC, χ, d * χ) * V1 * Diagonal(S)))
    dU1 .-= U1 * Diagonal(im .* imag.(vec(sum(dU1; dims = 1))))
    dV2 = V2 * (F .* (x -> x .+ x')(Diagonal(S) * U2' * reshape(dAC, χ * d, χ) * V2))
    dV2 .-= V2 * Diagonal(im .* imag.(vec(sum(dV2; dims = 1))))
    x = vcat(real.(diag(U1' * reshape(dAC, χ, d * χ) * V1 .- U2' * reshape(dAC, χ * d, χ) * V2)), real.(vec(sum(dU1; dims = 1))), real.(vec(sum(dV2; dims = 1))))
    A = zeros(3χ, 3χ)
    function a(x)
        dac = reshape(U1 * ((x -> x .+ x')(F .* (U1' * transpose(reshape(repeat(x[χ + 1 : 2χ], χ), χ, χ)))) * Diagonal(S) .+ Diagonal(x[1 : χ])) * V1', χ, d, χ) .+ reshape(U2 * (Diagonal(S) * (x -> x .+ x')(F .* (V2' * transpose(reshape(repeat(x[2χ + 1 : end], χ), χ, χ)))) .- Diagonal(x[1 : χ])) * V2', χ, d, χ)
        du1 = U1 * (F .* (x -> x .+ x')(U1' * reshape(dac, χ, d * χ) * V1 * Diagonal(S)))
        du1 .-= U1 * Diagonal(im .* imag.(vec(sum(du1; dims = 1))))
        dv2 = V2 * (F .* (x -> x .+ x')(Diagonal(S) * U2' * reshape(dac, χ * d, χ) * V2))
        dv2 .-= V2 * Diagonal(im .* imag.(vec(sum(dv2; dims = 1))))
        vcat(real.(diag(U1' * reshape(dac, χ, d * χ) * V1 .- U2' * reshape(dac, χ * d, χ) * V2)), real.(vec(sum(du1; dims = 1))), real.(vec(sum(dv2; dims = 1))))
    end
    for i in 1 : 3χ
        A[:, i] .= a(Matrix{Float64}(I, 3χ, 3χ)[:, i])
    end
    U, s, V = svd(A)
    temp = V[:, 1 : end - 3] * (Diagonal(inv.(s[1 : end - 3])) * (U[:, 1 : end - 3]' * x))
    dAC .-= reshape(U1 * ((x -> x .+ x')(F .* (U1' * transpose(reshape(repeat(temp[χ + 1 : 2χ], χ), χ, χ)))) * Diagonal(S) .+ Diagonal(temp[1 : χ])) * V1', χ, d, χ) .+ reshape(U2 * (Diagonal(S) * (x -> x .+ x')(F .* (V2' * transpose(reshape(repeat(temp[2χ + 1 : end], χ), χ, χ)))) .- Diagonal(temp[1 : χ])) * V2', χ, d, χ)
    dAC .-= AC .* real(dot(AC, dAC))
end

function polar_projection(AC)
    χ, d, = size(AC)
    C = gauge_fixing(AC)
    L1, = polar(reshape(AC, χ * d, χ))
    L2, = polar(C)
    reshape(L1 * L2', χ, d, χ)
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
        function fg2!(F, G, x)
            val, (dx,) = withgradient(y -> local_energy(polar_projection(y), y, h), x)
            if G !== nothing
                G .= dx
            end
            if F !== nothing
                return val
            end
        end
        U, S, = svdfix(A.C; fix = :U)
        AL = ein"(ij, jkl), lm -> ikm"(U', A.AL, U)
        AC = ein"ijk, k -> ijk"(AL, S)
        res = optimize(Optim.only_fg!(fg2!), AC, LBFGS(manifold = UniformMPS()), Optim.Options(g_abstol = tol, allow_f_increases = true, iterations = iterations))

        AC = Optim.minimizer(res)
        C = gauge_fixing(AC)
        invC = inv(C)
        AL = ein"ijk, kl -> ijl"(AC, invC)
        AR = ein"ij, jkl -> ikl"(invC, AC)
        A = MixedCanonicalMPS(AL, AR, AC, C)
    end

    R = AL2R(A.AL, Matrix{eltype(A.AL)}(I, χ, χ))
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