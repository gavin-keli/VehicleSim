

#Written by Gavin, A star
manhattan(a::RoadSegment, b::RoadSegment) = sum(abs.(b.lane_boundaries[1].pt_a - a.lane_boundaries[1].pt_a))

# check out the children of each RoadSegment
function get_children(map, p)
    res = RoadSegment[]
    #println("p.children ", p.children)
    for child in p.children
        #println("map[child] ", map[child])
        push!(res, map[child])
    end
    return res
end

function a_star_solver(map, start, goal)
    current_get_children(state) = get_children(map, state)
    # Here you can use any of the exported search functions, they all share the same interface, but they won't use the heuristic and the cost
    return astar(current_get_children, start, goal, heuristic=manhattan)
end


#Written by Eamon, Moved from example_project.jl
function if_in_segments(seg, ego_location)
    lb_1 = seg.lane_boundaries[1]
    if length(seg.lane_boundaries) == 2
        lb_2 = seg.lane_boundaries[2]
    else
        lb_2 = seg.lane_boundaries[3]
    end

    pt_a = lb_1.pt_a
    pt_b = lb_1.pt_b
    pt_c = lb_2.pt_a
    pt_d = lb_2.pt_b

    curvature = lb_1.curvature
    curved = !isapprox(curvature, 0.0; atol=1e-6)
    delta = pt_b-pt_a
    delta2 = pt_d-pt_b
    if !curved
        pt = 0.25*(pt_a+pt_b+pt_c+pt_d)
        check = abs(pt[1] - ego_location[1])
        check2 = abs(pt[2] - ego_location[2])
        if delta[1] == 0    
            if check < abs(delta2[1]/2) 
                if check2 < abs(delta[2]/2) 
                    return true
                else
                    return false
                end
            else
                return false
            end
        elseif delta[2] == 0
            if check < abs(delta[1]/2) 
                if check2 < abs(delta2[2]/2) 
                    return true
                else
                    return false
                end
            else
                return false
            end
        end
    else
        rad = 1.0 / abs(curvature)
        dist = π*rad/2.0
        left = curvature > 0

        rad_1 = rad
        #@info "rad_1: $rad_1"
        rad_2 = abs(pt_d[1]-pt_c[1])
        #@info "rad_2: $rad_2"

        if left
            if sign(delta[1]) == sign(delta[2])
                center = pt_a + [0, delta[2]]
            else
                center = pt_a + [delta[1], 0]
            end
        else
            if sign(delta[1]) == sign(delta[2])
                center = pt_a + [delta[1], 0]
            else
                center = pt_a + [0, delta[2]]
            end
        end

        r = (ego_location[1] - center[1])*(ego_location[1] - center[1]) + (ego_location[2] - center[2])*(ego_location[2] - center[2])
        #@info "center: $center, r: $r"
        if r < rad_1*rad_1
            return false
        end
        if r > rad_2*rad_2
            return false
        end
        if r > rad_1*rad_1
            if rad_2*rad_2 < r
                if left
                    if sign(delta[1]) == sign(delta[2])
                        if ego_location[1] < center[1] 
                            if ego_location[2] > center[2]
                                return true
                            end
                        end
                    end
                else
                    if ego_location[1] < center[1] 
                        if ego_location[2] < center[2]
                            return true
                        end
                    end
                end
            else
                if sign(delta[1]) == sign(delta[2])
                    if ego_location[1] < center[1] 
                        if ego_location[2] > center[2]
                            return true
                        end
                    end
                else
                    if ego_location[1] > center[1]
                        if ego_location[2] > center[2]
                            return true
                        end
                    end
                end
            end
        end
    end
    return false
end