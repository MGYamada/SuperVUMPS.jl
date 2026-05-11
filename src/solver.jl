Base.@kwdef struct SVUMPSOptions{T}
    tol::T = 1e-8
    iterations::Int = 1000
    hamiltonian::Bool = false
end

struct SVUMPSResult{E, M, H, O}
    energy::E
    state::M
    hamiltonian::H
    optim_result::O
end

function svumps_result(h, A; options::SVUMPSOptions = SVUMPSOptions(), tol = options.tol, iterations = options.iterations, Hamiltonian = options.hamiltonian, hamiltonian = Hamiltonian, verbose = false)
    optim_result = nothing
    ChainRulesCore.ignore_derivatives() do
        χ, d, = size(A.AL)
        U, _, V = svd(A.C)
        AC = ein"ij, (jkl, lm) -> ikm"(U', A.AC, V)

        function fg!(F, G, x)
            val, (dx,) = withgradient(x) do ac
                l, = polar(reshape(ac, χ * d, χ))
                al = reshape(l, χ, d, χ)
                real(local_energy(al, ac, h))
            end
            if G !== nothing
                G .= dx
            end
            if F !== nothing
                return val
            end
        end
        optim_result = optimize(only_fg!(fg!), AC, LBFGS(manifold = UniformMPS()), Optim.Options(g_abstol = tol, allow_f_increases = true, iterations = iterations))

        AC .= Optim.minimizer(optim_result)
        L, = polar(reshape(AC, χ * d, χ))
        AL = reshape(L, χ, d, χ)
        C, R = polar(reshape(AC, χ, d * χ); rev = true)
        AR = reshape(R, χ, d, χ)
        A = MixedCanonicalMPS(AL, AR, AC, C)
    end

    E = real(local_energy(A.AL, A.AC, h))
    H = hamiltonian ? construct_hamiltonian(h, A, E; tol = tol) : nothing
    SVUMPSResult(E, A, H, optim_result)
end

function svumps(h, A; kwargs...)
    result = svumps_result(h, A; kwargs...)
    if result.hamiltonian === nothing
        result.energy, result.state
    else
        result.energy, result.state, result.hamiltonian...
    end
end
