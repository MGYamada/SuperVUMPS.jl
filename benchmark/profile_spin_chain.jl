using Profile
using Printf

using SuperVUMPS

include(joinpath(@__DIR__, "..", "test", "spin_models.jl"))
using .SpinModels

function parse_arg(name, default)
    prefix = "--$(name)="
    for arg in ARGS
        if startswith(arg, prefix)
            value = arg[length(prefix) + 1:end]
            return default isa AbstractString ? value : parse(typeof(default), value)
        end
    end
    default
end

function parse_model()
    model = parse_arg("model", "tfim")
    if model == "tfim"
        "transverse-field Ising", transverse_field_ising(; g = 1.0)
    elseif model == "heisenberg"
        "Heisenberg", heisenberg()
    else
        error("unknown model: $(model)")
    end
end

function main()
    name, h = parse_model()
    χ = parse_arg("chi", 8)
    iterations = parse_arg("iterations", 3)
    tol = parse_arg("tol", 1e-6)
    repeats = parse_arg("repeats", 5)

    @printf("Profiling %s: chi=%d iterations=%d tol=%g repeats=%d\n", name, χ, iterations, tol, repeats)
    A = canonicalMPS(ComplexF64, χ, 2)

    svumps(h, A; tol, iterations)
    Profile.clear()
    @profile begin
        for _ in 1:repeats
            svumps(h, A; tol, iterations)
        end
    end

    data = Profile.fetch()
    @printf("Captured %d instruction-pointer samples\n\n", length(data))
    Profile.print(format = :flat, sortedby = :count, maxdepth = 80, mincount = 2)
end

main()
