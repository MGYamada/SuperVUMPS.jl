using LinearAlgebra
using Test

using SuperVUMPS
using Zygote

include("spin_models.jl")
using .SpinModels

function assert_mps_shape(A, χ, d)
    @test size(A.AL) == (χ, d, χ)
    @test size(A.AR) == (χ, d, χ)
    @test size(A.AC) == (χ, d, χ)
    @test size(A.C) == (χ, χ)
end

@testset "spin model fixtures" begin
    for h in (transverse_field_ising(), heisenberg())
        @test size(h) == (2, 2, 2, 2)
        H = reshape(h, 4, 4)
        @test H ≈ H'
    end
end

@testset "canonical MPS" begin
    χ = 2
    d = 2
    A = canonicalMPS(ComplexF64, χ, d)
    assert_mps_shape(A, χ, d)
    @test isapprox(norm(A.C), 1; atol = 1e-10)
end

@testset "local energy" begin
    χ = 2
    d = 2
    A = canonicalMPS(ComplexF64, χ, d)

    for h in (transverse_field_ising(; g = 0.7), heisenberg(; hz = 0.2))
        E = local_energy(A.AL, A.AC, h)
        @test isfinite(real(E))
        @test isapprox(imag(E), 0; atol = 1e-8)
    end
end

@testset "svumps smoke tests" begin
    for h in (transverse_field_ising(; g = 1.0), heisenberg())
        χ = 2
        d = 2
        A = canonicalMPS(ComplexF64, χ, d)
        E, A2 = svumps(h, A; tol = 1e-5, iterations = 2, verbose = false)

        @test isfinite(E)
        assert_mps_shape(A2, χ, d)
        @test isfinite(real(local_energy(A2.AL, A2.AC, h)))
    end
end

@testset "Hellmann-Feynman gradient" begin
    χ = 2
    d = 2
    h = transverse_field_ising(; g = 0.9)
    A = canonicalMPS(ComplexF64, χ, d)
    kwargs = (; tol = 1e-5, iterations = 1, verbose = false)

    _, Aopt = svumps(h, A; kwargs...)
    actual = Zygote.gradient(x -> svumps(x, A; kwargs...)[1], h)[1]
    expected = Zygote.gradient(x -> real(local_energy(Aopt.AL, Aopt.AC, x)), h)[1]

    @test actual ≈ expected atol = 1e-8 rtol = 1e-8
end

@testset "Hamiltonian construction" begin
    χ = 2
    d = 2
    h = transverse_field_ising(; g = 1.0)
    A = canonicalMPS(ComplexF64, χ, d)
    E, A2, HAC, HC = svumps_hamiltonian(h, A; tol = 1e-5, iterations = 1, verbose = false)

    @test isfinite(E)
    assert_mps_shape(A2, χ, d)
    @test size(HAC) == (χ, d, χ, χ, d, χ)
    @test size(HC) == (χ, χ, χ, χ)
end
