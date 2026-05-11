using LinearAlgebra
using Random
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

function random_hermitian_tensor(T, d, n)
    dim = d^n
    H = randn(T, dim, dim)
    H = (H + H') ./ 2
    reshape(H, ntuple(_ -> d, 2n))
end

@testset "SuperVUMPS" begin
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

@testset "polar decomposition" begin
    A = randn(ComplexF64, 4, 3)
    Q, P = SuperVUMPS.polar(A)

    @test Q' * Q ≈ Matrix{ComplexF64}(I, 3, 3) atol = 1e-10
    @test P ≈ P' atol = 1e-10
    @test Q * P ≈ A atol = 1e-10

    B = randn(ComplexF64, 3, 4)
    P_rev, Q_rev = SuperVUMPS.polar(B; rev = true)

    @test Q_rev * Q_rev' ≈ Matrix{ComplexF64}(I, 3, 3) atol = 1e-10
    @test P_rev ≈ P_rev' atol = 1e-10
    @test P_rev * Q_rev ≈ B atol = 1e-10
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

@testset "local energy higher-body tensors" begin
    χ = 2
    d = 2
    A = canonicalMPS(ComplexF64, χ, d)

    for h in (random_hermitian_tensor(ComplexF64, d, 3), random_hermitian_tensor(ComplexF64, d, 4))
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
        E, A2 = svumps(h, A; tol = 1e-5, iterations = 2)

        @test isfinite(E)
        assert_mps_shape(A2, χ, d)
        @test isfinite(real(local_energy(A2.AL, A2.AC, h)))
    end
end

@testset "svumps result API" begin
    χ = 2
    d = 2
    h = transverse_field_ising(; g = 1.0)
    A = canonicalMPS(ComplexF64, χ, d)
    options = SVUMPSOptions(; tol = 1e-5, iterations = 1, hamiltonian = true)
    result = svumps_result(h, A; options)

    @test result isa SVUMPSResult
    @test isfinite(result.energy)
    assert_mps_shape(result.state, χ, d)
    @test result.hamiltonian !== nothing

    HAC, HC = result.hamiltonian
    HAC2, HC2 = construct_hamiltonian(h, result.state, result.energy; tol = options.tol)
    HAC3, HC3 = Hamiltonian_construction(h, result.state, result.energy; tol = options.tol)
    @test HAC ≈ HAC2
    @test HC ≈ HC2
    @test HAC ≈ HAC3
    @test HC ≈ HC3
end

@testset "svumps end-to-end chi=4" begin
    χ = 4
    d = 2
    cases = (
        ("transverse-field Ising", transverse_field_ising(; g = 1.0), -1.24),
        ("Heisenberg", heisenberg(), -0.42),
    )

    for (name, h, energy_bound) in cases
        @testset "$name" begin
            Random.seed!(1234)
            A0 = canonicalMPS(ComplexF64, χ, d)

            E0 = real(local_energy(A0.AL, A0.AC, h))
            E, A = svumps(h, A0; tol = 1e-6, iterations = 25)

            @test isfinite(E)
            @test E < E0
            @test E < energy_bound
            @test E ≈ real(local_energy(A.AL, A.AC, h)) atol = 1e-10

            assert_mps_shape(A, χ, d)
            @test isapprox(norm(A.C), 1; atol = 1e-8)

            left_AC = reshape(reshape(A.AL, χ * d, χ) * A.C, χ, d, χ)
            right_AC = reshape(A.C * reshape(A.AR, χ, d * χ), χ, d, χ)
            @test A.AC ≈ left_AC atol = 1e-8
            @test A.AC ≈ right_AC atol = 1e-8
        end
    end
end

@testset "Hellmann-Feynman gradient" begin
    χ = 2
    d = 2
    h = transverse_field_ising(; g = 0.9)
    A = canonicalMPS(ComplexF64, χ, d)
    kwargs = (; tol = 1e-5, iterations = 1)

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
    E, A2, HAC, HC = svumps(h, A; tol = 1e-5, iterations = 1, Hamiltonian = true)

    @test isfinite(E)
    assert_mps_shape(A2, χ, d)
    @test size(HAC) == (χ, d, χ, χ, d, χ)
    @test size(HC) == (χ, χ, χ, χ)
    @test reshape(HAC, χ * d * χ, χ * d * χ) ≈ reshape(HAC, χ * d * χ, χ * d * χ)' atol = 1e-8
    @test reshape(HC, χ * χ, χ * χ) ≈ reshape(HC, χ * χ, χ * χ)' atol = 1e-8
end
end
