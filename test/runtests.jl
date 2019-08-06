using DifferentiableNeuralComputer
import Flux, Zygote
using Test

using DifferentiableNeuralComputer: DNCLSTM

@test_broken 1==2

@testset "DNCLSTM" begin
    m = DNCLSTM(3, 5, 2)
    @test m[2].state == m[2].init
    @test size.(m((rand(5, 4), rand(3, 4)))) == ((5, 4), (3,4))
    @test m[2].state != m[2].init
end
