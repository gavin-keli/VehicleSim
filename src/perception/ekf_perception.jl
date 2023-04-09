using LinearAlgebra
using Random
using Rotations
using StaticArrays

include("../measurements.jl")

"""
Generate 8 points representing the edge of a vehicle (3D) for other vehicle (perception)

World/Map view
"""
function get_3d_bbox_corners_perception(position, box_size)
    T = get_body_transform_perception(position)
    corners = []
    for dx in [-box_size[1]/2, box_size[1]/2]
        for dy in [-box_size[2]/2, box_size[2]/2]
            for dz in [-box_size[3]/2, box_size[3]/2]
                push!(corners, T*[dx, dy, dz, 1])
            end
        end
    end
    corners
end

"""
other vehicle frame to world frame (perception)
"""
function get_body_transform_perception(loc)
    R = one(RotMatrix{3, Float64})
    [R loc]
end

"""
Unicycle model
"""

# 7-dims vector x = [p1, p2, v, θ, l, w, h]
# REMOVE 2-dims vector u = [a, w]
# 8-dims vector z = [y1-4, y5-8] two camera views

# TODO change Function jac_fx, h, jac_hx, REMOVE Function jac_fu
"""
P(Xk|Xk-1)
"""
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
P(Zk|Xk,X_ego)

Inputs are 7-dims vector Xk = [p1, p2, v, θ, l, w, h]
            6-dims vector X_ego = [p1, p2, p3, r, p, y]
Outputs are 4 points bounding_boxes (2D) / 8-dims vector z = [[y1-4], [y5-8]] two camera views
"""
function  h_preception(x, x_ego)
    # here p1 and p2 are the center of a vehicle (not the GPS module location)
    p1 = x[1]
    p2 = x[2]
    p3 = 2.645 # height z
    v = x[3]
    θ = x[4]
    l = x[5]
    w = x[6]
    h = x[7]
    position = [p1, p2, p3]
    box_size = [l, w, h]

    ego_position = x_ego[1:3]
    ego_orientation = x_ego[4:7] # quat

    focal_len = 0.01
    pixel_len = 0.001
    image_width = 640
    image_height = 480

    Z = []

    # convert to bounding_boxes points (2D)
    corners_body = get_3d_bbox_corners_perception(position, box_size) # 8 points for one other vehicle (3D)

    # camera len x axes -> z axes -> camera frame -> body frame -> world frame

    T_body_cam1 = get_cam_transform(1) # camera1 camera -> body rotation
    T_body_cam2 = get_cam_transform(2) # camera2 camera -> body rotation
    T_cam_camrot = get_rotated_camera_transform() # camera len face x axes -> face z axes

    T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot) # combine two transform together
    T_body_camrot2 = multiply_transforms(T_body_cam2, T_cam_camrot) # same above

    T_world_body = get_body_transform(ego_orientation, ego_position) # EGO frame -> world frame
    T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1) # camera1 -> world
    T_world_camrot2 = multiply_transforms(T_world_body, T_body_camrot2) # camera2 -> world
    T_camrot1_world = invert_transform(T_world_camrot1) # world -> camera1
    T_camrot2_world = invert_transform(T_world_camrot2) # world -> camera2

    for (camera_id, transform) in zip((1,2), (T_camrot1_world, T_camrot2_world))
        vehicle_corners = [transform * [pt;1] for pt in corners_body] # 8 points (3D) but on camera views
        left = image_width/2
        right = -image_width/2
        top = image_height/2
        bot = -image_height/2

        for corner in vehicle_corners
            if corner[3] < focal_len
                break
            end
            px = focal_len*corner[1]/corner[3]
            py = focal_len*corner[2]/corner[3]
            left = min(left, px)
            right = max(right, px)
            top = min(top, py)
            bot = max(bot, py)
            # pick 4 points out of 8, still 3D on camera views. pz = focal_len
        end

        top = convert_to_pixel(image_height, pixel_len, top) # top 0.00924121388699952 => 251
        push!(Z,top)
        bot = convert_to_pixel(image_height, pixel_len, bot)
        push!(Z,bot)
        left = convert_to_pixel(image_width, pixel_len, left)
        push!(Z,left)
        right = convert_to_pixel(image_width, pixel_len, right)
        push!(Z,right)

        #push!(Z, (top, left, bot, right))

    end
    return Z # Z = [320,320,240,240,321,321,241,241] # first vector for camera1; second for camera2
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
