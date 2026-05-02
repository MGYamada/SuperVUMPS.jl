module SpinModels

using LinearAlgebra

export transverse_field_ising, heisenberg

const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const I2 = Matrix{ComplexF64}(I, 2, 2)

two_site_tensor(H) = reshape(Matrix{ComplexF64}(H), 2, 2, 2, 2)

function transverse_field_ising(; J = 1.0, g = 1.0)
    H = -J * kron(σz, σz) - (g / 2) * (kron(σx, I2) + kron(I2, σx))
    two_site_tensor(H)
end

function heisenberg(; J = 1.0, hz = 0.0)
    Sx = σx / 2
    Sy = σy / 2
    Sz = σz / 2
    H = J * (kron(Sx, Sx) + kron(Sy, Sy) + kron(Sz, Sz))
    H -= (hz / 2) * (kron(σz, I2) + kron(I2, σz))
    two_site_tensor(H)
end

end
