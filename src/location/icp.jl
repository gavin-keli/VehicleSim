"""
Implement the Iterative Closest Point algorithm.

Let R,t define a transformation representing the initial guess
of the ego position and orientation w.r.t. the map.

R and t map lidar points in EGO frame to MAP frame. 

E.x. imagine there is a lidar point at a distance 3 meters from ego vehicle, at an angle of -π/4 from straight ahead (1:30 using clock notation)
if we think that the the ego vehicle is at location [3.0, -1.0] in the P1-P2 plane of the map frame, 
with a heading of θ = π/6 (from the positive P1 axis in direction of positive P2 axis), then 

q = 3*[cos(-π/4), sin(-π/4)] ≈ [2.12, -2.12] in EGO frame
q' = R * q + t ≈ [cos(π/6) -sin(π/6);  * q + [3;
                  sin(π/6) cos(π/6)]        -1]
   ≈ [5.90, -1.78] 

                 o

            p  
            |  ↖e
            |
            |
p2__________0

here, o is the lidar point, ↖e is the ego vehicle, both expressed in the map frame. 
Notice that o has a relative position of ≈ [2.12, -2.12] expressed w.r.t. ego frame (forward facing frame)

Notice that R = [cos(θ) -sin(θ); sin(θ) cos(θ)] and t = [p1, p2], where θ is the estimated heading of ego and 
p1,p2 is the estimated position of ego in the map frame.

This algorithm iterates between two main steps.

1. Associate points in 'pointcloud' to 'map_points' in the unlabeled map, using current guess of transform.
    ( for each point q in pointcloud, find the point in the hd_map which is closest to R*q+t.
2. Update transform parameters to minimize distance betwen pointlcoud points and their associated pairs in the hd map.

Repeat this process until convergence. At the end, return the final transform R,t.
This transform (or rather the inverse of this transform) can be used to transform the map features 
into the ego frame, for planning purposes. However, you don't need to worry about that.

"""
function iterative_closest_point(map_points, pointcloud, R, t; max_iters = 10, visualize=true)
    if visualize
        f = Figure()
        ax = Axis(f[1,1], aspect=DataAspect())

        lims = ( - 5, 45)
        xlims = lims
        ylims = lims
        xlims!(ax, xlims)
        ylims!(ax, ylims)
        Mpx = [map_points[i][1] for i = 1:length(map_points)]
        Mpy = [map_points[i][2] for i = 1:length(map_points)]
        
        scatter!(ax, Mpx, Mpy, color=:blue)
        scene = display(f)

        Qt = [R*pointcloud[i]+t for i =1:length(pointcloud)]
        Qtx = [Qt[i][1] for i = 1:length(Qt)]
        Qty = [Qt[i][2] for i = 1:length(Qt)]
        sc = scatter!(ax, Qtx, Qty, color=:red)
    end

    
    @assert length(pointcloud) ≤ length(map_points) 
    N = length(pointcloud)

    point_associations = Vector{Int}(1:N)

    not_converged = true
    iters = 0
    err = 0
    while not_converged
        num_changes = update_point_associations!(point_associations, pointcloud, map_points, R, t)
        R, t, err = update_point_transform!(point_associations, pointcloud, map_points, R, t)
        iters += 1
        not_converged = (num_changes > 0) && iters < max_iters
        if visualize
            Qt = [R*pointcloud[i]+t for i =1:length(pointcloud)]
            Qtx = [Qt[i][1] for i = 1:length(Qt)]
            Qty = [Qt[i][2] for i = 1:length(Qt)]
            delete!(ax, sc)
            sc = scatter!(ax, Qtx, Qty, color=:red)
            @infiltrate
        end
        println("err in the loop", err)
    end

    return (; R, t, err)
end

"""
point_associations[i] = j means that pointcloud[i] is paired with map_points[j]

Here point_associations is updated in-place.

Here we WILL allow multiple points to be associated to the same map_point. Some implementations will
not allow this, but it is fine for us.

This function returns num_changes, which is how many elements of point_associations are changed.

"""
function update_point_associations!(point_associations, pointcloud, map_points, R, t)
    num_changes = 0
    # point_associations[i] = something for each i
    for i in eachindex(point_associations)
        tmp_map_points = R * pointcloud[i] + t
        L2 = norm(tmp_map_points-map_points[1])
        point_associations[i] = 1

        for j in 2:length(map_points)
            tmp_L2 = norm(tmp_map_points-map_points[j])
            if tmp_L2 < L2
                num_changes += 1
                point_associations[i] = j
                L2 = tmp_L2
            end
        end
    end
    return num_changes
end

"""
This function updates R and t in place, to minimize 

∑ ||pᵢ - Rqᵢ - t||²2

where pᵢ is the point in map_points indicated by point_associations[i]
and qᵢ is the i-th point in pointcloud.

You will need to derive the closed-form solution to this problem. You can do this from scratch 
(I recommend you try before looking up resources), but the full approach is mostly given in 
the following reference. Note that some of the slides have some typos, but the hand-written
proof is correct. You will need to keep track of the fact that some notational differences
exist between what we call variables, etc. and what that author calls things.
See https://cs.gmu.edu/~kosecka/cs685/cs685-icp.pdf, slides 1-8.

Return err = 1/N ∑ ||pᵢ - R*qᵢ-t ||^2
"""
function update_point_transform!(point_associations, pointcloud, map_points, R, t)
    #R .= something (notice the .=, which is used to update the elements of R in-place
    #t .= something (notice the .=, which is used to update the elements of t in-place
    new_map_points = [map_points[point_associations[1]]]

    for i in 2:length(point_associations)
        append!(new_map_points, [map_points[point_associations[i]]])
    end
    #println("new_map_points", new_map_points)

    μ_p = mean(new_map_points, dims=1)
    μ_q = mean(pointcloud, dims=1)

    #println("μ_p", μ_p)
    #println("μ_q", μ_q)

    P_center = new_map_points - vec(repeat(μ_p, 1, length(new_map_points)))
    Q_center = pointcloud - vec(repeat(μ_q, 1, length(pointcloud)))

    #println("P_center", P_center)
    #println("Q_center", Q_center)

    global W = [0.0 0.0; 0.0 0.0]
    for i in eachindex(P_center)
        global W += P_center[i]*Q_center[i]'
        #println("W", W)
    end

    U, S, V = svd(W)

    #println("V*U'", V*U')
    #println("det(V*U')", det(V*U'))

    if det(V*U') > 0
        R = U*V'
    else
        R = U*[1.0 0.0;0.0 -1.0]*V'
    end

    #R = U*V'
    #R = U*[1.0 0.0;0.0 -1.0]*V'
    #R = V*[1.0 0.0;0.0 -1.0]*U'
    #R = V*U'
    #println("R",R)

    t = μ_p[1] - R*μ_q[1]
    #println("t",t)

    global err = 0.0
    for i in eachindex(pointcloud)
        #global err += norm(new_map_points[i] - (R*pointcloud[i]+t))
        global err += norm(new_map_points[i] - (R*pointcloud[i]+t))^2
    end

    err = err/length(pointcloud)
    
    return R, t, err
end


function test_ICP(; Nb = 70, visualization=false)
    map_points = readdlm("hd_map_unlabeled.csv", ',', Float64)
    #map_points = readdlm("hd_map_features.csv", ',', Float64)
    map_points = eachrow(map_points) |> collect # transform into list of 2d points
    # assume pointcloud is also a list of 2d points
    
    ground_truth_map = example_map()
    dθ = -π .+ 2*π*(1:Nb)./Nb
    for i = 1:10
        x = feasible_point(ground_truth_map)
        θ = rand()*2*π-π
        Rinv = [cos(θ) sin(θ); -sin(θ) cos(θ)]
        pointcloud = []
        for d in dθ
            pt, α = compute_range(x, θ+d, ground_truth_map.segments; α_max=7.0)
            if isinf(α)
                continue
            else
                push!(pointcloud, Rinv*(pt .- x)) # transform to EGO frame
            end
        end
        θ̂ = θ + randn()*0.2
        x̂ = x + randn(2)*1.5

        R0 = [cos(θ̂) -sin(θ̂); sin(θ̂) cos(θ̂)]
        t0 = Vector{Float64}(x̂)
        (; R, t, err) = iterative_closest_point(map_points, pointcloud, R0, t0)
        x_estimated = t
        θ_estimated = atan(R[2,1], R[1,1])
        println("original error: $(norm(x-x̂)), $(θ-θ̂)")
        println("output error: $(norm(x-x_estimated)), $(θ-θ_estimated)")
        #println("x true: $x, x est: $x_estimated")
        #println("θ true: $θ, θ est: $θ_estimated")
    end
end

        



