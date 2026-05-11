struct MixedCanonicalMPS{T <: Complex}
    AL::Array{T, 3}
    AR::Array{T, 3}
    AC::Array{T, 3}
    C::Matrix{T}
end

function rightorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-12, maxiter = 100, verbose = false, kwargs...)
    χ, d, = size(A)
    Abar = conj(A)
    _, vecs1 = eigsolve(C * C', 1, :LR; ishermitian = false, tol = 1e-2tol, verbosity = 0, kwargs...) do X
        ein"ijk, (ljm, mk) -> li"(Abar, A, X)
    end
    ρ = vecs1[1]
    U, S, = svd(ρ)
    C = U * Diagonal(sqrt.(S)) * U'
    C ./= norm(C)
    L, R = polar(reshape(reshape(A, χ * d, χ) * C, χ, d * χ); rev = true)
    AR = Array(reshape(R, χ, d, χ))
    δ = norm(C .- L)
    numiter = 0
    while δ > tol && numiter < maxiter
        ARbar = conj(AR)
        _, vecs = eigsolve(L, 1, :LR; ishermitian = false, tol = 1e-2δ, verbosity = 0, kwargs...) do X
            ein"ijk, (ljm, mk) -> li"(ARbar, A, X)
        end
        C = vecs[1]
        C, = polar(C; rev = true)
        L, R = polar(reshape(reshape(A, χ * d, χ) * C, χ, d * χ); rev = true)
        AR .= reshape(R, χ, d, χ)
        L ./= norm(L)
        δ = norm(C .- L)
        numiter += 1
    end
    L, AR
end

function canonicalMPS(T, χ, d; verbose = false, kwargs...)
    U, _, V = svd(randn(T, χ * d, χ))
    AL = Array(reshape(U * V', χ, d, χ))
    C, AR = rightorth(AL; kwargs...)
    AC = ein"ijk, kl -> ijl"(AL, C)
    MixedCanonicalMPS(AL, AR, AC, C)
end

conjugateMPS(A) = MixedCanonicalMPS(conj(A.AL), conj(A.AR), conj(A.AC), conj(A.C))
