struct UniformMPS <: Manifold end

complex_to_real(x) = cat(real(x), imag(x); dims = 3)
real_to_complex(x) = x[:, :, 1] .+ im .* x[:, :, 2]

function Optim.retract!(::UniformMPS, AC; tol = 1e-12)
    χ, d, = size(AC)
    L, C = polar(reshape(AC, χ * d, χ))
    AL = reshape(L, χ, d, χ)
    ALbar = conj(AL)
    _, vecs1 = eigsolve(C * C', 1, :LR; ishermitian = false, tol = 1e-2tol, verbosity = 0) do x
        ein"ijk, (ljm, mk) -> li"(ALbar, AL, x)
    end
    X = vecs1[1]
    U, S, = svd(X)
    C .= U * Diagonal(sqrt.(S)) * U'
    C ./= norm(C)
    AC .= ein"ijk, kl -> ijl"(AL, C)
    AC ./= norm(AC)
end

function Optim.project_tangent!(::UniformMPS, dAC, AC; tol = 1e-12)
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
    temp, = linsolve(complex_to_real(K1 .+ K1' .- (K2 .+ K2')); ishermitian = true, isposdef = true, tol = tol, verbosity = 0) do x
        h = real_to_complex(x)
        dac = reshape(U1 * (Diagonal(invsqrtS1) * (h .+ h') * Diagonal(sqrtS1)) * V1', χ, d, χ) .- reshape(U2 * (Diagonal(sqrtS2) * (h .+ h') * Diagonal(invsqrtS2)) * V2', χ, d, χ)
        k1 = Diagonal(invsqrtS1) * (U1' * reshape(dac, χ, d * χ) * V1) * Diagonal(sqrtS1)
        k2 = Diagonal(invsqrtS2) * (V2' * reshape(dac, χ * d, χ)' * U2) * Diagonal(sqrtS2)
        complex_to_real(k1 .+ k1' .- (k2 .+ k2'))
    end
    h = real_to_complex(temp)
    dAC .-= reshape(U1 * (Diagonal(invsqrtS1) * (h .+ h') * Diagonal(sqrtS1)) * V1', χ, d, χ) .- reshape(U2 * (Diagonal(sqrtS2) * (h .+ h') * Diagonal(invsqrtS2)) * V2', χ, d, χ)
    dAC .-= AC .* real(dot(AC, dAC))
end
