
function example_map()
    segments = [ [[0,0], [40,0]],
            [[40,0], [40,40]],
            [[40,40], [0,40]],
            [[0,40], [0,0]],
            [[10,0], [10,10]],
            [[0,20], [10,20]],
            [[10,26], [10,34]],
            [[10,26], [6,30]],
            [[10,26], [20,26]],
            [[16,34], [20,40]],
            [[16,6], [16,14]],
            [[20,12], [26,4]],
            [[26,4], [20,4]],
            [[30,10], [30,24]],
            [[30,24], [20,18]],
            [[30,36], [34,36]],
            [[34,36], [36,34]],
            [[36,20], [36,6]],
            [[15,23], [18,20]],
            [[18,20], [15,17]],
            [[15,17], [12,20]],
            [[24,30], [30,30]],
            [[30,30], [36,26]],
            [[36,26], [40,26]],
            [[4,14], [6,14]],
            [[4,6], [6,6]],
            ]
    (; segments, xlims=(0,40), ylims=(0,40)) 
end

function feasible_point(map)
    x = rand()*(map.xlims[2]-map.xlims[1]) + map.xlims[1]
    y = rand()*(map.ylims[2]-map.ylims[1]) + map.ylims[1]
    Point2([x,y])
end

function compute_range(pos, θ, M; α_max = 5.0, σ = 0.05, p_missed=0.1, p_scatter=0.01)
    c = cos(θ)
    s = sin(θ)
    α_min = Inf
    for seg ∈ M
        A = [c seg[2][1]-seg[1][1];
             s seg[2][2]-seg[1][2]]
        b = seg[2] - [pos[1],pos[2]]
        α = A\b
        if 0 ≤ α[2] ≤ 1 && α[1] ≥ 0 # beam intersects map segment
            α_range = α[1] + randn()*σ
            α_min = min(α_min, α_range)
        end
    end
    r = rand()
    if r < p_missed
        α_min = Inf
    elseif r < p_missed + p_scatter
        α_min = α_max * rand()
    end
    if α_min > α_max
        α_min = Inf
    end
    (Point2(pos[1]+α_min*c, pos[2]+α_min*s), α_min)
end

function get_motion(old, Δ, M; σ=[0.075, 0.075, 0.025])
    α_min = 1.0
    θ = old[3]
    pos_Δ = [cos(θ), sin(θ)]*Δ[1]
    θ_Δ = Δ[2]
    new = old + [pos_Δ; θ_Δ] + σ.*randn(3)
    for seg ∈ M
        a = seg[1]
        b = seg[2]
        A = [new[1:2]-old[1:2] a-b]
        if rank(A) == 2
            coefs = A\(a-old[1:2])
            if all( 0 .≤ coefs .≤ 1 )
                α_min = min(α_min, coefs[1]-0.1)
            end
        end
    end
    α_min = max(α_min, 0.0)
    Point2(old[1:2] + α_min*(new[1:2]-old[1:2])), new[3]
end

function move!(x, delta, beams, M, map_pts; gps_noise=[0.15,0.15,0.1])
    (p, θ) = x
    old_pos = [p[][1],p[][2], θ[]]
    p[], θ[] = get_motion(old_pos, delta, M)
    N = length(beams)
    dists = zeros(Float64, N)
    gps_error = randn(3) .* gps_noise
    for (n, beam) ∈ enumerate(beams)
        θb = θ[] + 2*π*n/N
        beam[], dist = compute_range(p[], θb, M)
        dists[n] = dist
        new_point = Point2f(p[][1]+gps_error[1]+cos(θb+gps_error[3])*dist, p[][2]+gps_error[2]+sin(θb+gps_error[3])*dist)
        any(isinf.(new_point)) && continue
        map_pts[] = push!(map_pts[], new_point)
    end
end

function manual_collection(; Nb = 50)
    map = example_map()
    f = Figure()
    ax_l = Axis(f[1,1], aspect=DataAspect())
    ax_r = Axis(f[1,2], aspect=DataAspect())
    xlims = map.xlims .+ (-5, 5)
    ylims = map.ylims .+ (-5, 5)
    xlims!(ax_l, xlims)
    ylims!(ax_l, ylims)
    xlims!(ax_r, (-5,5))
    ylims!(ax_r, (-5,5))

    M = map.segments
    
    for segment ∈ M 
        lines!(ax_l, [segment[1][1],segment[2][1]], [segment[1][2],segment[2][2]], color=:black, linewidth=5)
    end

    p = Observable(feasible_point(map))
    θ = Observable(rand()*2*pi-pi)
    x = (p, θ)

    map_points = Observable(Point2f[])

    scatter!(ax_l, map_points, color=:blue)

    beams = [Observable(Point2(0.0,0.0)) for _ in 1:Nb]
    for (n, beam) ∈ enumerate(beams)
        θb = θ[] + 2*π*n/Nb
        beam[], dist = compute_range(p[], θb, M)
    end

    get_relative_pt = (X,B,angle) -> begin
        t = angle - pi/2
        R = [cos(t) sin(t); -sin(t) cos(t)]
        d = [B[1]-X[1], B[2]-X[2]]
        Rd = R*d
        Point2(Rd[1], Rd[2])
    end
    get_absolute_pt = (X,angle, dist) -> begin
        Point2(Rd[1], Rd[2])
    end

    for beam ∈ beams
        relative_pt = lift(get_relative_pt, p, beam, θ)
        scatter!(ax_r, relative_pt, color=:black)
        line_x = lift(pt->[0.0, pt[1]], relative_pt)
        line_y = lift(pt->[0.0, pt[2]], relative_pt)
        lines!(ax_r, line_x, line_y, color=:red, linewidth=1)
    end

    scatter!(ax_l, p, markersize=20,color=:red)
    
    scene = display(f)
    glfw_window = GLMakie.to_native(scene)

    on(events(f).keyboardbutton) do event
        if event.action in (Keyboard.press, Keyboard.repeat)
            event.key == Keyboard.left   && move!(x,[0.0,+π/6],beams,M,map_points)
            event.key == Keyboard.up     && move!(x,[1.0,0.0],beams,M, map_points)
            event.key == Keyboard.right  && move!(x,[0.0,-π/6],beams,M, map_points)
            event.key == Keyboard.down   && move!(x,[-1.0,0.0],beams,M, map_points)
            if event.key == Keyboard.s
                subsample_and_save(map_points[]) 
                println("Map points saved! Window can now be closed.")
            end
        end
    end
    current_figure()
end

function subsample_and_save(map_points; N_max = 10_000)
    N = length(map_points)
    points = sample(map_points, min(N, N_max), replace=false)
    x = [p[1] for p in points]
    y = [p[2] for p in points]
    pts = [x y]
    writedlm("hd_map_unlabeled.csv", pts, ", ")
end

function save_map_features(line_points)
    x = [p[1] for p in line_points] 
    y = [p[2] for p in line_points]
    pts = [x y]
    writedlm("hd_map_features.csv", pts, ", ")
end

function label_generated_map()
    points = readdlm("hd_map_unlabeled.csv", ',', Float64)
    f = Figure()
    ax = Axis(f[1,1], aspect=DataAspect())
    lims = (minimum(points) - 5, maximum(points) + 5)
    xlims = lims
    ylims = lims
    xlims!(ax, xlims)
    ylims!(ax, ylims)
    scatter!(ax, points[:,1], points[:,2], color=:blue)
    scene = display(f)
    glfw_window = GLMakie.to_native(scene)
    map_segs = []
    plot_segs = []
    
    line_points = Observable(Point2f[])
    linesegments!(ax, line_points, color = :red, linewidth=10)

    deregister_interaction!(ax, :rectanglezoom)
    on(events(f).keyboardbutton) do event
        if event.action in (Keyboard.press,)
            if event.key == Keyboard.u && !isempty(line_points[])
                pop!(line_points[])
                pop!(line_points[])
                notify(line_points)
            end
            if event.key == Keyboard.s
                save_map_features(line_points[]) 
                println("Map features saved! Window can now be closed.")
            end
        end
    end
    
    on(events(f).mousebutton) do event
        if event.button == Mouse.left && event.action == Mouse.press
            start_pos = mouseposition(ax.scene)
            pt = Point2f(start_pos[1], start_pos[2])
            push!(line_points[], pt, pt)
            notify(line_points)
        end
    end

    on(events(f).mouseposition) do mp
        end_pos = mouseposition(ax.scene)
        mb = events(f).mousebutton[]
        if mb.button == Mouse.left && (mb.action == Mouse.press || mb.action == Mouse.repeat)
            line_points[][end] = Point2f(end_pos...)
            notify(line_points)
        end
    end

    #on(events(f).mousebutton) do event
    #    if event.button == Mouse.left
    #        if event.action == Mouse.press
    #            start_pos = mouseposition(ax.scene)
    #            cur_xdir[] = [start_pos[1], start_pos[1]]
    #            cur_ydir[] = [start_pos[2], start_pos[2]]
    #            return Consume(false)
    #        end
    #        if event.action == Mouse.repeat
    #            pos = mouseposition(ax.scene)
    #            println(pos)
    #            cur_xdir[][2] = pos[1]
    #            cur_ydir[][2] = pos[2]
    #            return Consume(false)
    #        end
    #        if event.action == Mouse.release
    #            end_pos = mouseposition(ax.scene)
    #            push!(map_segs, [start_pos, end_pos])
    #            l = lines!(ax, [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color=:red, linewidth=7)
    #            push!(plot_segs, l)
    #            cur_xdir[] = [0,0]
    #            cur_ydir[] = [0,0]
    #            return Consume(false)
    #        end
    #    end
    #end
end

    
