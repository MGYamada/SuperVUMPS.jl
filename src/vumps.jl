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
                invP = U * Diagonal(inv.(S)) * U' # fix later
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
            ΔAA = reshape(linsolve(x -> vec(P * reshape(x, d, d) .+ reshape(x, d, d) * P), vec(ΔP); ishermitian = false)[1], d, d) # fix later
            ΔA += (ΔAA .+ ΔAA') * A
            (ΔA,)
        end
    else
        d = size(A, 2)
        Q, P = U * V', V * Diagonal(S) * V'
        (Q, P), function (Δ)
            if Δ[1] !== nothing
                invP = V * Diagonal(inv.(S)) * V' # fix later
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
            ΔAA = reshape(linsolve(x -> vec(P * reshape(x, d, d) .+ reshape(x, d, d) * P), vec(ΔP); ishermitian = false)[1], d, d) # fix later
            ΔA += A * (ΔAA .+ ΔAA')
            (ΔA,)
        end
    end
end

function leftorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-14, maxiter = 100, kwargs...) # fix later
    vals1, vecs1 = eigsolve(C' * C, 1, :LR; ishermitian = false, tol = tol, kwargs...) do ρ
        ein"(ij, ikl), jkm -> lm"(ρ, conj.(A), A)
    end
    ρ = vecs1[1]
    ρ += ρ'
    ρ /= tr(ρ)
    U, S, V = svd(ρ)
    C = Diagonal(sqrt.(S)) * V'
    _, C = qrpos(C)

    D, d, = size(A)
    Q, R = qrpos(reshape(C * reshape(A, D, d * D), D * d, D))
    AL = reshape(Q, D, d, D)
    λ = norm(R)
    R /= λ
    numiter = 1
    while norm(C .- R) > tol && numiter < maxiter
        vals2, vecs2 = eigsolve(R, 1, :LR; ishermitian = false, tol = tol, kwargs...) do X
            ein"(ij, ikl), jkm -> lm"(X, conj.(AL), A)
        end
        C = vecs2[1]
        _, C = qrpos(C)
        Q, R = qrpos(reshape(C * reshape(A, D, d * D), D * d, D))
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
    Ctemp = copy(C)
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
    AC1 = reshape(AC, χ * d, χ)
    AC2 = Array(reshape(AC, χ, d * χ)')
    f(x) = (x -> vcat(real(vec(x)), imag(vec(x))))(AC1' * reshape(Complex.(x[:, :, :, 1], x[:, :, :, 2]), χ * d, χ) .+ reshape(Complex.(x[:, :, :, 1], x[:, :, :, 2]), χ * d, χ)' * AC1 .- AC2' * reshape(Complex.(x[:, :, :, 1], x[:, :, :, 2]), χ, d * χ)' .- reshape(Complex.(x[:, :, :, 1], x[:, :, :, 2]), χ, d * χ) * AC2)
    J, = jacobian(f, cat(real(dAC), imag(dAC), dims = 4))
    vals, vecs = eigen(J * J')
    invJJ = vecs * Diagonal(map(x -> abs(x) < 1e-12 ? zero(x) : inv(x), vals)) * vecs'
    dAC .-= (x -> Complex.(x[:, :, :, 1], x[:, :, :, 2]))(reshape(J' * (invJJ * (J * vec(cat(real(dAC), imag(dAC), dims = 4)))), χ, d, χ, 2))
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

function conjugateMPS(A)
    MixedCanonicalMPS(conj.(A.AL), conj.(A.AR), conj.(A.AC), conj.(A.C))
end

function local_energy(AC, χ, d, h::Array{T, 4}) where T # two-site local Hamiltonian
    L, = polar(reshape(AC, χ * d, χ))
    AL = reshape(L, χ, d, χ)
    real(ein"ijk, (klm, (jlno, (inp, pom))) -> "(conj.(AL), conj.(AC), h, AL, AC)[])
end

function svumps(h::Array{T}, A; tol = 1e-12, Niter = 1000, Hamiltonian = false) where T
    χ, d, = size(A.AL)
    Abar = conjugateMPS(A)
    U, P = polar(A.C)
    AC = ein"ij, jkl -> ikl"(P, A.AR) # polar gauge

    function fg!(F, G, x)
        val, (dx,) = withgradient(y -> local_energy(y, χ, d, h), x)
        if G !== nothing
            G .= dx
        end
        if F !== nothing
            return val
        end
    end
    res = optimize(Optim.only_fg!(fg!), AC, LBFGS(manifold = UniformMPS()), Optim.Options(f_tol = tol, allow_f_increases = true, iterations = Niter))

    AC .= Optim.minimizer(res)
    L, = polar(reshape(AC, χ * d, χ))
    C, R = polar(reshape(AC, χ, d * χ); rev = true)
    AL = reshape(L, χ, d, χ)
    AR = reshape(R, χ, d, χ)
    A = MixedCanonicalMPS(AL, AR, AC, C)
    Abar = conjugateMPS(A)

    E = real(ein"ijk, (klm, (jlno, (inp, pom))) -> "(Abar.AL, Abar.AC, h, A.AL, A.AC)[])
    if Hamiltonian
        hr = h .- E .* ein"ij, kl -> ikjl"(Matrix{Float64}(I, d, d), Matrix{Float64}(I, d, d))
        Lh = regularize_left(A.AL, Abar.AL, A.C, Abar.C, hr, χ; tol = 1e-2tol)
        Rh = regularize_right(A.AR, Abar.AR, A.C, Abar.C, hr, χ; tol = 1e-2tol)
        HL = ein"ijk, (jlno, inp) -> klpo"(Abar.AL, hr, A.AL)
        HC = ein"ijk, (lmn, (jmop, (ioq, rpn))) -> klqr"(Abar.AL, Abar.AR, hr, A.AL, A.AR)
        HR = ein"klm, (jlno, pom) -> jknp"(Abar.AR, hr, A.AR)
        HAC = ein"klpo, ij -> klipoj"(HL, Matrix{Float64}(I, χ, χ)) .+ ein"jknp, hi -> hjkinp"(HR, Matrix{Float64}(I, χ, χ)) .+
        ein"ij, kl, mn -> ikmjln"(Lh, Matrix{Float64}(I, d, d), Matrix{Float64}(I, χ, χ)) .+ ein"ij, kl, mn -> ikmjln"(Matrix{Float64}(I, χ, χ), Matrix{Float64}(I, d, d), Rh)
        HC_rtn = HC .+ ein"ij, kl -> ikjl"(Lh, Matrix{Float64}(I, χ, χ)) .+ ein"ij, kl -> ikjl"(Matrix{Float64}(I, χ, χ), Rh)
        E, A, HAC, HC_rtn
    else
        E, A
    end
end
