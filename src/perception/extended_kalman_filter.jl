using Infiltrator
using LinearAlgebra
using Random

"""
Unicycle model
"""

# 7-dims vector x = [p1, p2, v, θ, l, w, h]
# REMOVE 2-dims vector u = [a, w]
# 8-dims vector z = [y1-4, y5-8] two camera views

# TODO change Function jac_fx, h, jac_hx, REMOVE Function jac_fu

function f(x, Δ)
    p1 = x[1]
    p2 = x[2]
    v = x[3]
    θ = x[4]
    l = x[5]
    w = x[6]
    h = x[7]

    [p1+Δ*v*cos(θ), p2+Δ*v*sin(θ), v, θ, l, w, h]
end


"""
Jacobian of f with respect to x, evaluated at x,u,Δ.
"""
function jac_fx(x, u, Δ)
    v = x[3]
    θ = x[4]
    [1.0 0 Δ*cos(θ) -Δ*v*sin(θ);
     0 1.0 Δ*sin(θ) Δ*v*cos(θ);
     0 0 1.0 0;
     0 0 0 1]
end

"""
Non-standard measurement model. Can we extract state estimate from just this?
"""
function h(x)
    [x[3]*(x[1]+x[2]) + cos(x[4])*sin(x[4]),]
end

"""
Jacobian of h with respect to x, evaluated at x.
"""
function jac_hx(x)
    # make sure to return a 1x4 matrix (not a 4 dim vector or a 4x1 matrix)
    [x[3] x[3] (x[1]+x[2]) cos(x[4])^2-sin(x[4])^2;]
end


"""
Extended kalman filter implementation.

Assume that the 'true' physical update in the world is given by 

xₖ = f(xₖ₋₁, uₖ, Δ), where Δ is the time difference between times k and k-1.

Here, uₖ is the 'true' controls applied to the system. These controls can be assumed to be a random variable,
with probability distribution given by 𝒩 (mₖ, proc_cov) where mₖ is some IMU-like measurement, and proc_cov is a constant covariance matrix.

The process model distribution is then approximated as:

P(xₖ | xₖ₋₁, uₖ) ≈ 𝒩 ( Axₖ₋₁ + Buₖ + c, Σ̂ )

where 
A = ∇ₓf(μₖ₋₁, mₖ, Δ),
B = ∇ᵤf(μₖ₋₁, mₖ, Δ),
c = f(μₖ₋₁, mₖ, Δ) - Aμₖ₋₁ - Bmₖ

μ̂ = Aμₖ₋₁ + Bmₖ + c
  = f(μₖ₋₁, mₖ, Δ)
Σ̂ = A Σₖ₋₁ A' + B proc_cov B', 


Further, assume that the 'true' measurement generation in the world is given by

zₖ = h(xₖ) + wₖ,

where wₖ is some additive gaussian noise with probability density function given by

𝒩 (0, meas_var).

The measurement model is then approximated as 

P(zₖ | xₖ) ≈ 𝒩 ( C xₖ + d , meas_var )


where 
C = ∇ₓ h(μ̂), 
d = h(μ̂) - Cμ̂

The extended Kalman filter update equations can be implemented as the following:

Σₖ = (Σ̂⁻¹ + C' (meas_var)⁻¹ C)⁻¹
μₖ = Σₖ ( Σ̂⁻¹ μ̂ + C' (meas_var)⁻¹ (zₖ - d) )

"""
function filter(; μ=zeros(4), Σ=Diagonal([5,5,3,1.0]), x0=zeros(4), num_steps=25, meas_freq=0.5, meas_jitter=0.025, meas_var=Diagonal([0.25,]), proc_cov = Diagonal([0.2, 0.1]), rng=MersenneTwister(5), output=true)
    gt_states = [x0,] # ground truth states that we will try to estimate
    timesteps = []
    u_constant = randn(rng) * [5.0, 0.2]
    μs = [μ,]
    Σs = Matrix{Float64}[Σ,]
    zs = Vector{Float64}[]

    u_prev = zeros(2)
    x_prev = x0

    for k = 1:num_steps
        uₖ = u_constant
        mₖ = uₖ + sqrt(proc_cov) * randn(rng, 2) # Noisy IMU measurement.
        Δ = meas_freq + meas_jitter * (2*rand(rng) - 1)
        xₖ = f(x_prev, uₖ, Δ)
        x_prev = xₖ
        u_prev = uₖ
        zₖ = h(xₖ) + sqrt(meas_var) * randn(rng, 1)
        
        A = jac_fx(x_prev, mₖ, Δ)
        B = jac_fu(x_prev, mₖ, Δ)

        μ̂ = f(μ, mₖ, Δ)
        Σ̂ = A*Σ*A' + B*proc_cov*B'

        C = jac_hx(μ̂)
        d = h(μ̂) - C*μ̂
        
        Σ = inv(inv(Σ̂) + C'*inv(meas_var)*C)
        μ = Σ * (inv(Σ̂) * μ̂ + C'*inv(meas_var) * (zₖ - d))
         
        push!(μs, μ)
        push!(Σs, Σ)
        push!(zs, zₖ)
        push!(gt_states, xₖ)
        push!(timesteps, Δ)
        if output
            println("Ttimestep ", k, ":")
            println("   Ground truth (x,y): ", xₖ[1:2])
            println("   Estimated (x,y): ", μ[1:2])
            println("   Ground truth v: ", xₖ[3])
            println("   estimated v: ", μ[3])
            println("   Ground truth θ: ", xₖ[4])
            println("   estimated θ: ", μ[4])
            println("   measurement received: ", zₖ)
            println("   Uncertainty measure (det(cov)): ", det(Σ))
        end
    end

    (; μs, Σs)
end

"""
Inputs are 4 points bounding_boxes (2D) / 8-dims vector z = [y1-4, y5-8] two camera views
Outputs are 8 points (3D) maybe
"""
function  inverse_cameras(vehicles, state_channels, cam_channels; max_rate=10.0, focal_len = 0.01, pixel_len = 0.001, image_width = 640, image_height = 480)
    
end
