using Test
using ekf_perception
using Random

@testset "derivative tests" begin
    rng = MersenneTwister(1)
    for i = 1:10
        x = randn(rng, 4)
        u = randn(rng, 2)
        Δ = 0.1

        xx = HW4.f(x,u,Δ)
        Jx = HW4.jac_fx(x,u,Δ)
        Ju = HW4.jac_fu(x,u,Δ)
        Jxn = similar(Jx)
        Jun = similar(Ju)
        for j = 1:4
            ej = zeros(4); ej[j]=1e-6;
            Jxn[:,j] = (HW4.f(x+ej,u,Δ)-xx) / 1e-6
        end
        for j = 1:2
            ej = zeros(2); ej[j]=1e-6;
            Jun[:,j] = (HW4.f(x,u+ej,Δ)-xx) / 1e-6
        end
        zz = HW4.h(x)
        Jxh = HW4.jac_hx(x)
        Jxhn = similar(Jxh)
        for j = 1:4
            ej = zeros(4); ej[j]=1e-6;
            Jxhn[:,j] = (HW4.h(x+ej)-zz) / 1e-6
        end
        @test isapprox(Jx, Jxn; atol=1e-6)
        @test isapprox(Ju, Jun; atol=1e-6)
        @test isapprox(Jxh, Jxhn; atol=1e-6)
    end
end

@testset "filter tests" begin
    rng = MersenneTwister(1)
    (; μs, Σs) = HW4.filter(;rng, num_steps=5, output=false)
    @test μs[end] ≈ [4.469684109807531, -0.5074208788587384, 3.8334776235079744, 0.013346243056787443]
    @test Σs[end] ≈ [4.083422649885581 -4.760820675538723 0.6783058545436464 -0.3859568477084995; -4.760820675538731 5.783629691211686 -1.0168986150730464 0.5736488493415263; 0.6783058545436517 -1.0168986150730503 0.3464750263607838 -0.1915152895406309; -0.38595684770850314 0.5736488493415293 -0.19151528954063124 0.13499480889667884]
    
    rng = MersenneTwister(2)
    (; μs, Σs) = HW4.filter(;rng, num_steps=5, output=false)
    @test μs[end] ≈ [9.066811252283514, 1.3690868205449078, 9.646816851469339, 0.4164551969297463]
    @test Σs[end] ≈ [9.344102085398303 -11.223827612687634 1.7556376939231129 -0.7533837088382146; -11.223827612687561 13.673210393250809 -2.2860937542266697 0.9806170922781743; 1.7556376939230405 -2.2860937542265973 0.4963117074406993 -0.21309909769353946; -0.7533837088381826 0.9806170922781412 -0.21309909769353907 0.11447736733417953]
    
    rng = MersenneTwister(3)
    (; μs, Σs) = HW4.filter(;rng, num_steps=5, output=false)
    @test μs[end] ≈ [12.85230957926251, 5.638895771000534, 15.281288208934711, 0.7402617202023976]
    @test Σs[end] ≈ [46.87352518607232 -55.71410192641209 6.480427835956306 -3.3056207382927187; -55.714101926413285 66.35406729681806 -7.799042093483749 3.9819663193693553; 6.480427835957187 -7.799042093484628 0.9669804715504098 -0.4950287745728834; -3.305620738293179 3.981966319369818 -0.49502877457288585 0.27970063920732946]
end
