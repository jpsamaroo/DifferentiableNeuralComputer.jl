using DifferentiableNeuralComputer
import Flux, Zygote
using Test

using DifferentiableNeuralComputer: DNCLstm

@test_broken 1==2

begin
    m = DNCLstm(3, 5)
    @test m.state == m.init
    @test size.(m(rand(5, 4), rand(3, 4))) == ((5, 4), (3,4))
    @test m.state != m.init
end
