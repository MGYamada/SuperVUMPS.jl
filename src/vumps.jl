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

function gauge_fixing(AC, u, v)
    χ, d, = size(AC)
    U, S1, = svdfix(reshape(AC, χ, d * χ); fix = :U)
    _, S2, V = svdfix(reshape(AC, χ * d, χ); fix = :V)
    U = U * Diagonal(map(x -> x / abs(x), u))
    V = V * Diagonal(map(x -> x / abs(x), v))
    U, ((S1 .+ S2) ./ 2), V
end

function svdfix(A; fix = :U)
    U, S, V = svd(A)
    if fix == :U
        phase = map(x -> x / abs(x), diag(U))
        U .= U * Diagonal(phase)'
        V .= V * Diagonal(phase)'
    elseif fix == :V
        phase = map(x -> x / abs(x), diag(V))
        U .= U * Diagonal(phase)'
        V .= V * Diagonal(phase)'
    end
    U, S, V
end

Zygote.@adjoint svdfix(A; kwargs...) = svdfix_back(A; kwargs...)

function svdfix_back(A; η = 1e-40, kwargs...)
    U, S, V = svdfix(A; kwargs...)
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

struct UniformMPS <: Manifold
    χ::Int
    d::Int
end

function Optim.retract!(mfd::UniformMPS, x)
    χ, d = mfd.χ, mfd.d
    AC = reshape(x[1 : end - 2χ], χ, d, χ)
    u = x[end - 2χ + 1 : end - χ]
    v = x[end - χ + 1 : end]
    U, S, V = gauge_fixing(AC, u, v)
    C = U * Diagonal(S) * V'
    invC = V * Diagonal(inv.(S)) * U'
    AL = ein"ijk, kl -> ijl"(AC, invC)
    AR = ein"ij, jkl -> ikl"(invC, AC)
    AL, L, = leftorth(AL)
    R, AR, = rightorth(AR)
    # AC .= ein"(ij, jkl), lm -> ikm"(L, AC, R)
    C .= L * C * R
    C ./= norm(C)
    U, S, = svd(C)
    AL = ein"(ij, jkl), lm -> ikm"(U', AL, U)
    AC = ein"ijk, k -> ijk"(AL, S)
    AC ./= norm(AC)
    x .= vcat(vec(AC), ones(eltype(AC), 2χ))
end

function Optim.project_tangent!(mfd::UniformMPS, dx, x; η = 1e-40)
    # Currently metric gauge-dependent
    χ, d = mfd.χ, mfd.d
    AC = reshape(x[1 : end - 2χ], χ, d, χ)
    u = x[end - 2χ + 1 : end - χ]
    v = x[end - χ + 1 : end]
    U1, S, V2 = gauge_fixing(AC, u, v)
    U, _, V1 = svd(U1' * reshape(AC, χ, d * χ))
    V1 .= V1 * U'
    U2, _, V = svd(reshape(AC, χ * d, χ) * V2)
    U2 .= U2 * V'
    dAC = reshape(dx[1 : end - 2χ], χ, d, χ)
    du = dx[end - 2χ + 1 : end - χ]
    dv = dx[end - χ + 1 : end]
    S² = S .^ 2
    F = S²' .- S²
    @. F /= (F ^ 2 + η)
    dU1 = U1 * (F .* (x -> x .+ x')(U1' * reshape(dAC, χ, d * χ) * V1 * Diagonal(S)))
    dV2 = V2 * (F .* (x -> x .+ x')(Diagonal(S) * U2' * reshape(dAC, χ * d, χ) * V2))
    temp, = linsolve(vcat(diag(U1' * reshape(dAC, χ, d * χ) * V1 .- U2' * reshape(dAC, χ * d, χ) * V2), diag(dU1) .- du, diag(dV2) .- dv)) do x
        dac = reshape(U1 * (Diagonal(x[1 : χ]) .+ (x -> x .+ x')(F .* (U1' * Diagonal(x[χ + 1 : 2χ]))) * Diagonal(S)) * V1', χ, d, χ)
        .- reshape(U2 * (Diagonal(x[1 : χ]) .- Diagonal(S) * (x -> x .+ x')(F .* (V2' * Diagonal(x[2χ + 1 : end])))) * V2', χ, d, χ)
        duu = -x[χ + 1 : 2χ]
        dvv = -x[2χ + 1 : end]
        du1 = U1 * (F .* (x -> x .+ x')(U1' * reshape(dac, χ, d * χ) * V1 * Diagonal(S)))
        dv2 = V2 * (F .* (x -> x .+ x')(Diagonal(S) * U2' * reshape(dac, χ * d, χ) * V2))
        vcat(diag(U1' * reshape(dac, χ, d * χ) * V1 .- U2' * reshape(dac, χ * d, χ) * V2), diag(du1) .- duu, diag(dv2) .- dvv)
    end
    dAC .-= reshape(U1 * (Diagonal(temp[1 : χ]) .+ (x -> x .+ x')(F .* (U1' * Diagonal(temp[χ + 1 : 2χ]))) * Diagonal(S)) * V1', χ, d, χ)
    .- reshape(U2 * (Diagonal(temp[1 : χ]) .- Diagonal(S) * (x -> x .+ x')(F .* (V2' * Diagonal(temp[2χ + 1 : end])))) * V2', χ, d, χ)
    du .+= temp[χ + 1 : 2χ]
    dv .+= temp[2χ + 1 : end]
    dAC .-= AC .* real(dot(AC, dAC))
    dx[1 : end - 2χ] .= vec(dAC)
    dx[end - 2χ + 1 : end - χ] .= du
    dx[end - χ + 1 : end] .= dv
    dx
end

function polar_projection(AC, u, v)
    χ, d, = size(AC)
    U, _, V = gauge_fixing(AC, u, v)
    AC2 = ein"ijk, kl -> ijl"(AC, V * U')
    L, = polar(reshape(AC2, χ * d, χ))
    reshape(L, χ, d, χ)
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
        val, (dx,) = withgradient(y -> local_energy(polar_projection(reshape(y[1 : end - 2χ], χ, d, χ), y[end - 2χ + 1 : end - χ], y[end - χ + 1 : end]), reshape(y[1 : end - 2χ], χ, d, χ), h), x)
        if G !== nothing
            G .= dx
        end
        if F !== nothing
            return val
        end
    end
    U, S, = svd(A.C)
    AL = ein"(ij, jkl), lm -> ikm"(U', A.AL, U)
    AC = ein"ijk, k -> ijk"(AL, S)
    res = optimize(Optim.only_fg!(fg!), vcat(vec(AC), ones(eltype(AC), 2χ)), LBFGS(linesearch = BackTracking(), manifold = UniformMPS(χ, d)), Optim.Options(g_abstol = tol, allow_f_increases = true, iterations = iterations))

    y = Optim.minimizer(res)
    AC = reshape(y[1 : end - 2χ], χ, d, χ)
    AL = polar_projection(reshape(y[1 : end - 2χ], χ, d, χ), y[end - 2χ + 1 : end - χ], y[end - χ + 1 : end])
    # AC = y[:, 1 : end - 1, :]
    # C = y[:, end, :]
    # AL = AC2AL(y)
    # AR = AC2AR(y)
    # A = MixedCanonicalMPS(AL, AR, AC, C)

    E = local_energy(AL, AC, h)
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