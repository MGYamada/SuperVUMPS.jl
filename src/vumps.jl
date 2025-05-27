function polar(A; rev = false)
    U, S, V = svd(A)
    if rev
        U * Diagonal(S) * U', U * V'
    else
        U * V', V * Diagonal(S) * V'
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

function rightvect(AL)
    C, = rightorth(AL)
    C * C'
end

Zygote.@adjoint function rightvect(AL)
    R = rightvect(AL)
    R, function (Δ)
        ALbar = conj.(AL)
        ξ, = linsolve(x -> ein"(ij, ikl), jkm -> lm"(x, ALbar, AL) .- x, Δ - I * dot(R, Δ))
        _, back = pullback(x -> ein"ijk, ljm -> ilkm"(x, conj(x)), AL)
        back(-ein"ij, kl -> ijkl"(ξ, conj.(R)))
    end
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

local_energy(AL, R, h::Array{T, 4}) where T = real(ein"ijk, (klm, (jlno, (inp, (poq, qm)))) -> "(conj.(AL), conj.(AL), h, AL, AL, R)[])

function Hamiltonian_construction(h::Array{T, 4}, A, E; tol = 1e-12) where T # fix later
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

function svumps(h::T, AL; tol = 1e-12, Niter = 1000, Hamiltonian = false) where T
    χ, d, = size(AL)

    function fg!(F, G, x)
        val, (dx,) = withgradient(y -> local_energy(reshape(y, χ, d, χ), rightvect(reshape(y, χ, d, χ)), h), x)
        if G !== nothing
            G .= dx
        end
        if F !== nothing
            return val
        end
    end
    res = optimize(Optim.only_fg!(fg!), reshape(AL, χ * d, χ), LBFGS(manifold = Stiefel()), Optim.Options(f_tol = tol, allow_f_increases = true, iterations = Niter))

    AL .= reshape(Optim.minimizer(res), χ, d, χ)
    E = local_energy(AL, rightvect(AL), h)
    if Hamiltonian
        E, AL, Hamiltonian_construction(h, AL, E; tol = tol)...
    else
        E, AL
    end
end

Zygote.@adjoint function svumps(h, AL; kwargs...)
    X = svumps(h, AL; kwargs...)
    _, AL, = X
    X, function (Δ)
        if all(Δ[2 : end] .=== nothing)
            R = rightvect(AL)
            _, back = pullback(x -> local_energy(AL, R, x), h)
            (back(Δ[1])[1], nothing)
        else
            error("MPS/effective Hamiltonian differentiation not supported")
        end
    end
end