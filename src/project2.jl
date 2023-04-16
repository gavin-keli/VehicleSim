struct SimpleVehicleState
    p1::Float64
    p2::Float64
    θ::Float64
    v::Float64
    l::Float64
    w::Float64
    h::Float64
end

struct FullVehicleState
    position::SVector{3, Float64}
    orientation::SVector{4, Float64} # still quat
    velocity::SVector{3, Float64}
    angular_vel::SVector{3, Float64}
end

struct MyLocalizationType
    last_update::Float64
    x::FullVehicleState
end

struct MyPerceptionType
    last_update::Float64
    x::Vector{SimpleVehicleState}
end

function localize(gps_channel, imu_channel, localization_state_channel)
    # Set up algorithm / initialize variables
    while true
        fresh_gps_meas = []
        while isready(gps_channel)
            meas = take!(gps_channel)
            push!(fresh_gps_meas, meas)
        end
        fresh_imu_meas = []
        while isready(imu_channel)
            meas = take!(imu_channel)
            push!(fresh_imu_meas, meas)
        end
        
        # process measurements

        localization_state = MyLocalizationType(0,0.0)
        if isready(localization_state_channel)
            take!(localization_state_channel)
        end
        put!(localization_state_channel, localization_state)
    end 
end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel)
    # set up stuff
    while true
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        latest_localization_state = fetch(localization_state_channel)
        
        # process bounding boxes / run ekf / do what you think is good

        perception_state = MyPerceptionType(0,0.0)
        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
    end
end

function decision_making(localization_state_channel, 
        perception_state_channel, 
        map, 
        target_road_segment_id, 
        socket)
    # do some setup
    while true
        latest_localization_state = fetch(localization_state_channel)
        latest_perception_state = fetch(perception_state_channel)

        # figure out what to do ... setup motion planning problem etc
        steering_angle = 0.0
        target_vel = 3.0
        cmd = VehicleCommand(steering_angle, target_vel, true)
        serialize(socket, cmd)
    end
end

function decision_making_2(gt_channel, 
    map, 
    target_channel, 
    socket)

    # do some setup
    gt_vehicle_states = []
    current_segment = map[1]
    current_position = [0.0, 0.0]
    target_road_segment_id = 101

    while true
        @info "begining decision_making_2"
        #latest_localization_state = fetch(localization_state_channel)
        #latest_perception_state = fetch(perception_state_channel)

        if isready(target_channel)
            target_road_segment_id = fetch(target_channel)
        end

        while isready(gt_channel)
            meas = take!(gt_channel)
            gt_vehicle_states = meas
            @info "updated"
        end
        sleep(1)

        @info gt_vehicle_states

        if gt_vehicle_states != []
            current_position = gt_vehicle_states.position[1:2]
        end

        @info "searching current segment"
        @info current_position

        # search all map_segments
        for (key,value) in map
            if if_in_segments(map[key], current_position)
                current_segment = map[key]
                @info "current segment: $current_segment"
            end
        end

        @info "found segment"
        @info "current segment"
        @info current_segment
        @info "target segment"
        @info map[target_road_segment_id]

        # path finding A_star
        res = a_star_solver(map, current_segment, map[target_road_segment_id])
        #@info res
        ####### print out the whole path from start point to end point ######
        for i in res.path
            print(i.id)
            println(i.children)
        end

        @info "in the decision_making_2"
        steering_angle = current_segment.lane_boundaries[1].curvature
        target_vel = 3.0
        cmd = VehicleCommand(steering_angle, target_vel, true)
        serialize(socket, cmd)
        @info "end decision_making_2"
    end
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end


function project_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.training_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    localization_state_channel = Channel{MyLocalizationType}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)
    target_channel = Channel(1)

    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        sleep(0.001)
        local measurement_msg
        #measurement_msg = deserialize(socket)

        received = false
        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
                received = true
            else
                break
            end
        end
        !received && continue

        target_map_segment = measurement_msg.target_segment
        if !isfull(target_channel)
            put!(target_channel, target_map_segment)
        end
        ego_vehicle_id = measurement_msg.vehicle_id

        for meas in measurement_msg.measurements
            if meas isa GPSMeasurement
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)

    #@async localize(gps_channel, imu_channel, localization_state_channel)
    #@async perception(cam_channel, localization_state_channel, perception_state_channel)
    #@async decision_making(localization_state_channel, perception_state_channel, map_segments, target_map_segment, socket)
    @async decision_making_2(gt_channel ,map_segments, target_channel, socket)
end