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

safesign(x::Number) = iszero(x) ? one(x) : sign(x)

qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    Q, R
end

lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    F = qr!(Matrix(A'))
    Q = Matrix(Matrix(F.Q)')
    L = Matrix(F.R')
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    L, Q
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
        real.(AL), real.(R), λ
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
        real.(L), real.(AR), λ
    else
        L, AR, λ
    end
end

struct UniformMPS <: Manifold end

function Optim.retract!(::UniformMPS, x; tol = 1e-12)
    _, d, = size(x)
    d -= 1
    AC = x[:, 1 : d, :]
    C = x[:, end, :]
    U, S, V = svd(C)
    Cinv = V * Diagonal(inv.(S)) * U'
    AL = ein"ijk, kl -> ijl"(AC, Cinv)
    _, L, = leftorth(AL; tol = tol)
    AR = ein"ij, jkl -> ikl"(Cinv, AC)
    R, = rightorth(AR; tol = tol)
    AC .= ein"ij, (jkl, lm) -> ikm"(L, AC, R)
    C .= L * C * R
    AC ./= norm(AC)
    C ./= norm(C)
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
    U0, S0, V0 = svd(C)
    U1, S1, V1 = svd(U0' * reshape(AC, χ, d * χ))
    V1 .= V1 * U1'
    U1 .= U0
    U2, S2, V2 = svd(reshape(AC, χ * d, χ) * V0)
    U2 .= U2 * V2'
    V2 .= V0
    sqrtS0 = sqrt.(S0)
    invsqrtS0 = inv.(sqrtS0)
    sqrtS1 = sqrt.(S1)
    invsqrtS1 = inv.(sqrtS1)
    sqrtS2 = sqrt.(S2)
    invsqrtS2 = inv.(sqrtS2)
    K01 = Diagonal(invsqrtS0) * (U0' * dC * V0) * Diagonal(sqrtS0)
    K02 = Diagonal(invsqrtS0) * (V0' * dC' * U0) * Diagonal(sqrtS0)
    K1 = Diagonal(invsqrtS1) * (U1' * reshape(dAC, χ, d * χ) * V1) * Diagonal(sqrtS1)
    K2 = Diagonal(invsqrtS2) * (V2' * reshape(dAC, χ * d, χ)' * U2) * Diagonal(sqrtS2)
    temp, = linsolve((x -> cat(real(x), imag(x); dims = 4))(cat(K01 .+ K01' .- (K1 .+ K1'), K02 .+ K02' .- (K2 .+ K2'); dims = 3)); ishermitian = true, isposdef = true, tol = tol, verbosity = 0) do x
        h1 = x[:, :, 1, 1] .+ im .* x[:, :, 1, 2]
        h2 = x[:, :, 2, 1] .+ im .* x[:, :, 2, 2]
        dc = U0 * (Diagonal(invsqrtS0) * (h1 .+ h1') * Diagonal(sqrtS0) .+ Diagonal(sqrtS0) * (h2 .+ h2') * Diagonal(invsqrtS0)) * V0' 
        dac = -(reshape(U1 * (Diagonal(invsqrtS1) * (h1 .+ h1') * Diagonal(sqrtS1)) * V1', χ, d, χ) .+ reshape(U2 * (Diagonal(sqrtS2) * (h2 .+ h2') * Diagonal(invsqrtS2)) * V2', χ, d, χ))
        k01 = Diagonal(invsqrtS0) * (U0' * dc * V0) * Diagonal(sqrtS0)
        k02 = Diagonal(invsqrtS0) * (V0' * dc' * U0) * Diagonal(sqrtS0)
        k1 = Diagonal(invsqrtS1) * (U1' * reshape(dac, χ, d * χ) * V1) * Diagonal(sqrtS1)
        k2 = Diagonal(invsqrtS2) * (V2' * reshape(dac, χ * d, χ)' * U2) * Diagonal(sqrtS2)
        (x -> cat(real(x), imag(x); dims = 4))(cat(k01 .+ k01' .- (k1 .+ k1'), k02 .+ k02' .- (k2 .+ k2'); dims = 3))
    end
    h1 = temp[:, :, 1, 1] .+ im .* temp[:, :, 1, 2]
    h2 = temp[:, :, 2, 1] .+ im .* temp[:, :, 2, 2]
    dC .-= U0 * (Diagonal(invsqrtS0) * (h1 .+ h1') * Diagonal(sqrtS0) .+ Diagonal(sqrtS0) * (h2 .+ h2') * Diagonal(invsqrtS0)) * V0' 
    dAC .+= reshape(U1 * (Diagonal(invsqrtS1) * (h1 .+ h1') * Diagonal(sqrtS1)) * V1', χ, d, χ) .+ reshape(U2 * (Diagonal(sqrtS2) * (h2 .+ h2') * Diagonal(invsqrtS2)) * V2', χ, d, χ)
    dC .-= C .* real(dot(C, dC))
    dAC .-= AC .* real(dot(AC, dAC))
    dx[:, 1 : d, :] .= dAC
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
    res = optimize(Optim.only_fg!(fg!), cat(A.AC, reshape(A.C, χ, 1, χ); dims = 2), LBFGS(manifold = UniformMPS()), Optim.Options(g_abstol = tol, allow_f_increases = true, iterations = iterations))

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