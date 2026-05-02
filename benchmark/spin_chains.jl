using LinearAlgebra
using Printf
using Random

using SuperVUMPS

include(joinpath(@__DIR__, "..", "test", "spin_models.jl"))
using .SpinModels

function parse_arg(name, default)
    prefix = "--$(name)="
    for arg in ARGS
        startswith(arg, prefix) && return parse(typeof(default), arg[length(prefix) + 1:end])
    end
    default
end

function median_time(f, repeats)
    times = Vector{Float64}(undef, repeats)
    for i in eachindex(times)
        GC.gc()
        times[i] = @elapsed f()
    end
    sort!(times)
    times[cld(length(times), 2)]
end

function run_case(name, h, exact_energy; χ, iterations, tol, repeats, verbose)
    Random.seed!(1234)
    A0 = canonicalMPS(ComplexF64, χ, 2; verbose)

    E, A = svumps(h, A0; tol, iterations, verbose)
    energy = real(local_energy(A.AL, A.AC, h))
    @printf("%-24s E = %.12f check = %.12f error = %.6e\n", name, E, energy, abs(E - exact_energy))

    GC.gc()
    elapsed = median_time(repeats) do
        svumps(h, A0; tol, iterations, verbose)
    end

    GC.gc()
    allocated = @allocated svumps(h, A0; tol, iterations, verbose = false)
    @printf("%-24s time = %.6f s allocated = %.3f MiB\n", name, elapsed, allocated / 2.0^20)
end

function main()
    χ = parse_arg("chi", 4)
    iterations = parse_arg("iterations", 10)
    tol = parse_arg("tol", 1e-8)
    verbose = parse_arg("verbose", false)
    repeats = parse_arg("repeats", 3)

    @printf("SuperVUMPS spin-chain benchmark: chi=%d iterations=%d tol=%g repeats=%d verbose=%s\n", χ, iterations, tol, repeats, verbose)
    run_case("transverse-field Ising", transverse_field_ising(; g = 1.0), -4 / pi; χ, iterations, tol, repeats, verbose)
    run_case("Heisenberg", heisenberg(), 1 / 4 - log(2); χ, iterations, tol, repeats, verbose)
end

main()
