using LinearAlgebra
using Random
using Rotations
using StaticArrays
using ForwardDiff

include("../measurements.jl")

"""
P(Xk|Xk-1)

13-dims vector x = [position, quaternion, velocity, angular_vel]
8-dims vector z = [y1-4, y5-8] two camera views

 Can be used as process model for EKF
 which estimates xₖ = [position; quaternion; velocity; angular_vel]
 We haven't discussed quaternions in class much, but interfacing with GPS/IMU
 will be much easier using this representation, which is used internally by the simulator.
 """
function f_localization(position, quaternion, velocity, angular_vel, Δt)
    r = angular_vel
    mag = norm(r)

    sᵣ = cos(mag*Δt / 2.0)
    vᵣ = sin(mag*Δt / 2.0) * r/mag

    sₙ = quaternion[1]
    vₙ = quaternion[2:4]

    s = sₙ*sᵣ - vₙ'*vᵣ
    v = sₙ*vᵣ+sᵣ*vₙ+vₙ×vᵣ

    new_position = position + Δt * velocity
    new_quaternion = [s; v]
    new_velocity = velocity
    new_angular_vel = angular_vel
    return [new_position; new_quaternion; new_velocity; new_angular_vel]
end


"""
Jacobian of f_localization with respect to x, evaluated at x,Δ.
"""
function jac_fx_localization(position, quaternion, velocity, angular_vel, Δt)
    r = angular_vel
    mag = norm(r)

    sᵣ = cos(mag*Δt / 2.0)
    vᵣ = sin(mag*Δt / 2.0) * r/mag

    sₙ = quaternion[1]
    vₙ = quaternion[2:4]

    s = sₙ*sᵣ - vₙ'*vᵣ
    v = sₙ*vᵣ+sᵣ*vₙ+vₙ×vᵣ

    # Some results I'm not sure about.
    # For Column 11
    jf11a = (4*(r[1]*conj(r[1]))^(1/2)*mag^2)
    jf11b = (2*(r[1]*conj(r[1]))^(1/2)*mag^2^(3/2))
    jf11c = abs(r[1])*(r[1] + conj(r[1]))

    jf_4_11 = (vᵣ[1]*mag*abs(r[1])*conj(vₙ[1])*(r[1] + conj(r[1])))/(2*(r[1]*conj(r[1]))^(1/2)*mag^2^(3/2)) - (sₙ*Δt*sin((Δt*mag)/2)*abs(r[1])*(r[1] + conj(r[1])))/(4*(r[1]*conj(r[1]))^(1/2)*mag) - (sin((Δt*mag)/2)*conj(vₙ[1]))/mag + (vᵣ[2]*mag*abs(r[1])*conj(vₙ[2])*(r[1] + conj(r[1])))/(2*(r[1]*conj(r[1]))^(1/2)*mag^2^(3/2)) + (vᵣ[3]*mag*abs(r[1])*conj(vₙ[3])*(r[1] + conj(r[1])))/(2*(r[1]*conj(r[1]))^(1/2)*mag^2^(3/2)) - (r[1]*Δt*sᵣ*abs(r[1])*conj(vₙ[1])*(r[1] + conj(r[1])))/(4*(r[1]*conj(r[1]))^(1/2)*mag^2) - (r[2]*Δt*sᵣ*abs(r[1])*conj(vₙ[2])*(r[1] + conj(r[1])))/(4*(r[1]*conj(r[1]))^(1/2)*mag^2) - (r[3]*Δt*sᵣ*abs(r[1])*conj(vₙ[3])*(r[1] + conj(r[1])))/(4*(r[1]*conj(r[1]))^(1/2)*mag^2)
    jf_5_11 = (sₙ*sin((Δt*mag)/2))/mag - (vₙ[1]*Δt*sin((Δt*mag)/2)*jf11c)/(4*(r[1]*conj(r[1]))^(1/2)*mag) - (r[1]*sₙ*sin((Δt*mag)/2)*jf11c)/jf11b + (r[2]*vₙ[3]*sin((Δt*mag)/2)*jf11c)/jf11b - (r[3]*vₙ[2]*sin((Δt*mag)/2)*jf11c)/jf11b + (r[1]*sₙ*Δt*sᵣ*jf11c)/jf11a - (r[2]*vₙ[3]*Δt*sᵣ*jf11c)/jf11a + (r[3]*vₙ[2]*Δt*sᵣ*jf11c)/jf11a
    jf_6_11 = (vₙ[3]*sin((Δt*mag)/2))/mag - (vₙ[2]*Δt*sin((Δt*mag)/2)*jf11c)/(4*(r[1]*conj(r[1]))^(1/2)*mag) - (r[2]*sₙ*sin((Δt*mag)/2)*jf11c)/jf11b - (r[1]*vₙ[3]*sin((Δt*mag)/2)*jf11c)/jf11b + (r[3]*vₙ[1]*sin((Δt*mag)/2)*jf11c)/jf11b + (r[2]*sₙ*Δt*sᵣ*jf11c)/jf11a + (r[1]*vₙ[3]*Δt*sᵣ*jf11c)/jf11a - (r[3]*vₙ[1]*Δt*sᵣ*jf11c)/jf11a
    jf_7_11 = (r[1]*vₙ[2]*sin((Δt*mag)/2)*jf11c)/jf11b - (vₙ[3]*Δt*sin((Δt*mag)/2)*jf11c)/(4*(r[1]*conj(r[1]))^(1/2)*mag) - (vₙ[2]*sin((Δt*mag)/2))/mag - (r[2]*vₙ[1]*sin((Δt*mag)/2)*jf11c)/jf11b - (r[3]*sₙ*sin((Δt*mag)/2)*jf11c)/jf11b - (r[1]*vₙ[2]*Δt*sᵣ*jf11c)/jf11a + (r[2]*vₙ[1]*Δt*sᵣ*jf11c)/jf11a + (r[3]*sₙ*Δt*sᵣ*jf11c)/jf11a

    # For Column 12
    jf12a = (4*(r[2]*conj(r[2]))^(1/2)*mag^2)
    jf12b = (2*(r[2]*conj(r[2]))^(1/2)*mag^2^(3/2))
    jf12c = abs(r[2])*(r[2] + conj(r[2]))

    jf_4_12 = (vᵣ[1]*mag*abs(r[2])*conj(vₙ[1])*(r[2] + conj(r[2])))/jf12b - (sₙ*Δt*sin((Δt*mag)/2)*jf12c)/(4*(r[2]*conj(r[2]))^(1/2)*mag) - (sin((Δt*mag)/2)*conj(vₙ[2]))/mag + (vᵣ[2]*mag*abs(r[2])*conj(vₙ[2])*(r[2] + conj(r[2])))/jf12b + (vᵣ[3]*mag*abs(r[2])*conj(vₙ[3])*(r[2] + conj(r[2])))/jf12b - (r[1]*Δt*sᵣ*abs(r[2])*conj(vₙ[1])*(r[2] + conj(r[2])))/jf12a - (r[2]*Δt*sᵣ*abs(r[2])*conj(vₙ[2])*(r[2] + conj(r[2])))/jf12a - (r[3]*Δt*sᵣ*abs(r[2])*conj(vₙ[3])*(r[2] + conj(r[2])))/jf12a
    jf_5_12 = (r[2]*vₙ[3]*sin((Δt*mag)/2)*jf12c)/jf12b - (vₙ[1]*Δt*sin((Δt*mag)/2)*jf12c)/(4*(r[2]*conj(r[2]))^(1/2)*mag) - (r[1]*sₙ*sin((Δt*mag)/2)*jf12c)/jf12b - (vₙ[3]*sin((Δt*mag)/2))/mag - (r[3]*vₙ[2]*sin((Δt*mag)/2)*jf12c)/jf12b + (r[1]*sₙ*Δt*sᵣ*jf12c)/jf12a - (r[2]*vₙ[3]*Δt*sᵣ*jf12c)/jf12a + (r[3]*vₙ[2]*Δt*sᵣ*jf12c)/jf12a
    jf_6_12 = (sₙ*sin((Δt*mag)/2))/mag - (vₙ[2]*Δt*sin((Δt*mag)/2)*jf12c)/(4*(r[2]*conj(r[2]))^(1/2)*mag) - (r[2]*sₙ*sin((Δt*mag)/2)*jf12c)/jf12b - (r[1]*vₙ[3]*sin((Δt*mag)/2)*jf12c)/jf12b + (r[3]*vₙ[1]*sin((Δt*mag)/2)*jf12c)/jf12b + (r[2]*sₙ*Δt*sᵣ*jf12c)/jf12a + (r[1]*vₙ[3]*Δt*sᵣ*jf12c)/jf12a - (r[3]*vₙ[1]*Δt*sᵣ*jf12c)/jf12a
    jf_7_12 = (vₙ[1]*sin((Δt*mag)/2))/mag - (vₙ[3]*Δt*sin((Δt*mag)/2)*jf12c)/(4*(r[2]*conj(r[2]))^(1/2)*mag) + (r[1]*vₙ[2]*sin((Δt*mag)/2)*jf12c)/jf12b - (r[2]*vₙ[1]*sin((Δt*mag)/2)*jf12c)/jf12b - (r[3]*sₙ*sin((Δt*mag)/2)*jf12c)/jf12b - (r[1]*vₙ[2]*Δt*sᵣ*jf12c)/jf12a + (r[2]*vₙ[1]*Δt*sᵣ*jf12c)/jf12a + (r[3]*sₙ*Δt*sᵣ*jf12c)/jf12a

    # For Column 13
    jf13a = (4*(r[3]*conj(r[3]))^(1/2)*mag^2)
    jf13b = (2*(r[3]*conj(r[3]))^(1/2)*mag^2^(3/2))
    jf13c = abs(r[3])*(r[3] + conj(r[3]))

    jf_4_13 = (vᵣ[1]*mag*abs(r[3])*conj(vₙ[1])*(r[3] + conj(r[3])))/jf13b - (sₙ*Δt*sin((Δt*mag)/2)*jf13c)/(4*(r[3]*conj(r[3]))^(1/2)*mag) - (sin((Δt*mag)/2)*conj(vₙ[3]))/mag + (vᵣ[2]*mag*abs(r[3])*conj(vₙ[2])*(r[3] + conj(r[3])))/jf13b + (vᵣ[3]*mag*abs(r[3])*conj(vₙ[3])*(r[3] + conj(r[3])))/jf13b - (r[1]*Δt*sᵣ*abs(r[3])*conj(vₙ[1])*(r[3] + conj(r[3])))/jf13a - (r[2]*Δt*sᵣ*abs(r[3])*conj(vₙ[2])*(r[3] + conj(r[3])))/jf13a - (r[3]*Δt*sᵣ*abs(r[3])*conj(vₙ[3])*(r[3] + conj(r[3])))/jf13a
    jf_5_13 = (vₙ[2]*sin((Δt*mag)/2))/mag - (vₙ[1]*Δt*sin((Δt*mag)/2)*jf13c)/(4*(r[3]*conj(r[3]))^(1/2)*mag) - (r[1]*sₙ*sin((Δt*mag)/2)*jf13c)/jf13b + (r[2]*vₙ[3]*sin((Δt*mag)/2)*jf13c)/jf13b - (r[3]*vₙ[2]*sin((Δt*mag)/2)*jf13c)/jf13b + (r[1]*sₙ*Δt*sᵣ*jf13c)/jf13a - (r[2]*vₙ[3]*Δt*sᵣ*jf13c)/jf13a + (r[3]*vₙ[2]*Δt*sᵣ*jf13c)/jf13a
    jf_6_13 = (r[3]*vₙ[1]*sin((Δt*mag)/2)*jf13c)/jf13b - (vₙ[2]*Δt*sin((Δt*mag)/2)*jf13c)/(4*(r[3]*conj(r[3]))^(1/2)*mag) - (r[2]*sₙ*sin((Δt*mag)/2)*jf13c)/jf13b - (r[1]*vₙ[3]*sin((Δt*mag)/2)*jf13c)/jf13b - (vₙ[1]*sin((Δt*mag)/2))/mag + (r[2]*sₙ*Δt*sᵣ*jf13c)/jf13a + (r[1]*vₙ[3]*Δt*sᵣ*jf13c)/jf13a - (r[3]*vₙ[1]*Δt*sᵣ*jf13c)/jf13a
    jf_7_13 = (sₙ*sin((Δt*mag)/2))/mag - (vₙ[3]*Δt*sin((Δt*mag)/2)*jf13c)/(4*(r[3]*conj(r[3]))^(1/2)*mag) + (r[1]*vₙ[2]*sin((Δt*mag)/2)*jf13c)/jf13b - (r[2]*vₙ[1]*sin((Δt*mag)/2)*jf13c)/jf13b - (r[3]*sₙ*sin((Δt*mag)/2)*jf13c)/jf13b - (r[1]*vₙ[2]*Δt*sᵣ*jf13c)/jf13a + (r[2]*vₙ[1]*Δt*sᵣ*jf13c)/jf13a + (r[3]*sₙ*Δt*sᵣ*jf13c)/jf13a


    [1. 0. 0.   0.      0.      0.      0.      Δt 0. 0. 0.         0.          0.;
     0. 1. 0.   0.      0.      0.      0.      0. Δt 0. 0.         0.          0.;
     0. 0. 1.   0.      0.      0.      0.      0. 0. Δt 0.         0.          0.;
     0. 0. 0.   sᵣ      -vᵣ[1]  -vᵣ[2]  -vᵣ[3]  0. 0. 0. jf_4_11    jf_4_12     jf_4_13;
     0. 0. 0.   vᵣ[1]   sᵣ      vᵣ[3]   -vᵣ[2]  0. 0. 0. jf_5_11    jf_5_12     jf_5_13;
     0. 0. 0.   vᵣ[2]   -vᵣ[3]  sᵣ      vᵣ[1]   0. 0. 0. jf_6_11    jf_6_12     jf_6_13;
     0. 0. 0.   vᵣ[3]   vᵣ[2]   -vᵣ[1]  sᵣ      0. 0. 0. jf_7_11    jf_7_12     jf_7_13;
     0. 0. 0.   0.      0.      0.      0.      1. 0. 0. 0.         0.          0.;
     0. 0. 0.   0.      0.      0.      0.      0. 1. 0. 0.         0.          0.;
     0. 0. 0.   0.      0.      0.      0.      0. 0. 1. 0.         0.          0.;
     0. 0. 0.   0.      0.      0.      0.      0. 0. 0. 1.         0.          0.;
     0. 0. 0.   0.      0.      0.      0.      0. 0. 0. 0.         1.          0.;
     0. 0. 0.   0.      0.      0.      0.      0. 0. 0. 0.         0.          1.]

end

"""
P(Zk|Xk)

Inputs are 13-dims vector Xk = [position, quaternion, velocity, angular vel]
Outputs are 8-dims vector Zk = [[xy], [v], [w]]
"""
function  h_localization(x)

    T = get_gps_transform()
    gps_loc_body = T*[zeros(3); 1.0]

    xyz_body = x[1:3]
    q_body = x[4:7]
    velocity = x[8: 10]
    angular_vel = [11:13]

    Tbody = get_body_transform(q_body, xyz_body)
    xyz_gps = Tbody * [gps_loc_body; 1.0]
    meas = xyz_gps[1:2] 

    # Do Imu now
    T_body_imu = get_imu_transform()
    T_imu_body = invert_transform(T_body_imu)
    R_imu = T_imu_body[1:3,1:3]
    p_imu = T_imu_body[1:3,end]

    w_imu = R_imu * angular_vel
    v_imu = R_imu * velocity + p_imu × w_imu

    return [meas, v_imu, w_imu]
end

"""
Jacobian of h with respect to x, evaluated at x.
"""
function jac_hx_localization(x)
    [1.0 0.0 0.0    (26*x[6])/5-6*x[4]-2*x[7]       2*x[6]-6*x[5]+(26*x[7])/5       (26*x[4])/5+2*x[5]+6*x[6]       (26*x[5])/5-2*x[4]+6*x[7]       0.0     0.0     0.0     0.0     0.0     0.0;
     0.0 1.0 0.0    2*x[4]-(26*x[5])/5-6*x[7]       -(26*x[4])/5-2*x[5]-6*x[6]      2*x[6]-6*x[5]+(26*x[7])/5       (26*x[6])/5-6*x[4]-2*x[7]       0.0     0.0     0.0     0.0     0.0     0.0;
     0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.9998  0.0     -0.0199 0.0     0.7     0.0;
     0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.0     1.0     0.0     -0.7    0.0     0.0;
     0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.0199  0.0     0.9998  0.0     0.014   0.0;
     0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.0     0.0     0.0     0.9998  0.0     -0.0199;
     0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.0     0.0     0.0     0.0     1.0     0.0;
     0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.0     0.0     0.0     0.0199  0.0     0.9998]
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
function ekf_perception(; μ=zeros(4), Σ=Diagonal([5,5,3,1.0]), x0=zeros(4), num_steps=25, meas_freq=0.5, meas_jitter=0.025, meas_var=Diagonal([0.25,]), proc_cov = Diagonal([0.2, 0.1]), rng=MersenneTwister(5), output=true)
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
