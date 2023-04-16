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
        # meas can be considered as Zk
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        # an example of meas in the fresh_cam_meas
        # VehicleSim.CameraMeasurement(1.680665593489583e9, 1, 0.01, 0.001, 640, 480, SVector{4, Int64}[a,b,c,d])



        latest_localization_state = fetch(localization_state_channel)
        # an example of latest_localization_state
        # MyLocalizationType(1.6666, FullVehicleState([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]))
        
        # here are some info about EGO vehicle, those can be used to decide the Xk/X0 of other vehicles
        position = latest_localization_state.x.position
        velocity = latest_localization_state.x.velocity
        orientation = latest_localization_state.x.orientation
        angular_vel = latest_localization_state.x.angular_vel

        
        # process bounding boxes / run ekf / do what you think is good

        # Test 1 (1 vehicle, 1 EGO)
        # run ekf when there are bounding_boxes in the lstest_cam_meas_channel
        # end ekf when no bounding_boxes in the cam_meas_channel
        # CALL EKF Here
        
        while length(fetch(fresh_cam_meas).bounding_boxes) != 0
            (; Xk, Σs) = ekf_perception()            
        end

        t = time()
        perception_state = MyPerceptionType(t,[Xk])

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
    @info "into decision making"

    t = time()
    while true
        #@info "into decision making while loop"
        #latest_localization_state = fetch(localization_state_channel)
        #latest_perception_state = fetch(perception_state_channel)
        
        gt_vehicle_states = []

        if !isready(localization_state_channel)
            continue
        end

        while isready(localization_state_channel)
            meas = take!(localization_state_channel)
            id = meas.vehicle_id
            gt_vehicle_states = meas
            @info "updated"
        end

        @info gt_vehicle_states

        latest_true_ego_state = gt_vehicle_states

        @info "latest_true_ego_state"
        @info latest_true_ego_state

        #if isready(localization_state_channel)
        #    @info "localization_state_channel is ready"
        #    latest_localization_state = take!(localization_state_channel)
        #    @info "latest_localization_state"
        #    @info latest_localization_state
        #    sleep(1)
        #else
        #    @info "localization_state_channel is not ready"
        #    sleep(1)
        #    continue
        #end
        #latest_perception_state = take!(perception_state_channel)

        # figure out the current segments
        current_segment = 0
        current_position = [0.0, 0.0]

        #current_position = latest_localization_state.x.position[1:2]
        #current_position = latest_localization_state.position[1:2]
        if length(latest_true_ego_state) != 0
            current_position = latest_true_ego_state.position[1:2]
        end

        @info "searching current segment"
        @info current_position

        # search all map_segments
        for (key,value) in map
            if if_in_segments(map[key], current_position)
                current_segment = map[key]
                #@info "current segment: $current_segment"
            end
        end

        @info "found segment"
        @info "current segment: $current_segment"
        #@info "target segment: $map[target_road_segment_id]"

        # path finding A_star
        #res = a_star_solver(map, current_segment, map[target_road_segment_id])

        @info "found path"

        # figure out what to do ... setup motion planning problem etc
        steering_angle = current_segment.lane_boundaries[1].curvature
        target_vel = 3.0
        cmd = VehicleCommand(steering_angle, target_vel, true)
        serialize(socket, cmd)
    end
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end


function project_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = training_map()
    msg = deserialize(socket) # Visualization info

    @info "msg"
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    #localization_state_channel = Channel{MyLocalizationType}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)

    @info "channels created"

    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    @async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        local measurement_msg

        @info "into while loop"

        #measurement_msg = deserialize(socket)

        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
            else
                break
            end
        end

        target_map_segment = measurement_msg.target_segment
        ego_vehicle_id = measurement_msg.vehicle_id

        @info "measurement_msg"
        #@info measurement_msg

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

        @info "put all channels"
        @info fetch(gt_channel)

        # put the gt_channel into two channels, localization_state_channel, and perception_state_channel 
        # for test purpose only
        # MyLocalizationType meas
        #meas = take!(gt_channel)
        #@info "gt_channel"
        #@info meas

        
        #@info "capture gt_channel meas"
        #localization_meas  = MyLocalizationType(meas.time, FullVehicleState(meas.position, #meas.velocity, meas.orientation, meas.angular_velocity))
        #if isready(localization_state_channel)
        #    @info "localization_state_channel is full"
        #    take!(localization_state_channel)
        #    put!(localization_state_channel, localization_meas)
        #else
        #    @info "insert localization_meas"
        #    put!(localization_state_channel, localization_meas)
        #end
        # MyPerceptionType meas
        perception_meas = MyPerceptionType(meas.time,[SimpleVehicleState(0.0, 0.0, 0.0, 0.0, 13.2, 5.7, 5.3)])
        if !isfull(perception_state_channel)
            put!(perception_state_channel, perception_meas)
        end
        @info "end of first loop"
    end

    @async while true
        decision_making(gt_channel, perception_state_channel, map_segments, target_map_segment, socket)
    end

    #@async localize(gps_channel, imu_channel, localization_state_channel)
    #@async perception(cam_channel, localization_state_channel, perception_state_channel)
    #@async decision_making(localization_state_channel, perception_state_channel, map_segments, target_map_segment, socket)
    #@async decision_making(localization_state_channel, perception_state_channel, map_segments, target_map_segment, socket)
    decision_making(gt_channel, perception_state_channel, map_segments, target_map_segment, socket)
    #decision_making(localization_state_channel, perception_state_channel, map_segments, target_map_segment, socket)

end
