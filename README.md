# VehicleSim
VehicleSim

# Loading / instantiating code

```julia
(VehicleSim) pkg> instantiate
(VehicleSim) pkg> add https://github.com/forrestlaine/MeshCat.jl
(VehicleSim) pkg> add https://github.com/forrestlaine/RigidBodyDynamics.jl

julia> using VehicleSim, Sockets
```

# Running Simulation
```julia
julia> s = server();
[ Info: Server can be connected to at 1.2.3.4 and port 4444
[ Info: Server visualizer can be connected to at 1.2.3.4:8712
```

This will spin up the server / simulation engine. For now, the server will instantiate a single vehicle. 

# Connecting a keyboard client

```julia
julia> keyboard_client(ip"1.2.3.4")
[ Info: Client accepted.
[ Info: Client follow-cam can be connected to at 1.2.3.4:8713
[ Info: Press 'q' at any time to terminate vehicle.
```

# Shutting down server
```julia
julia> shutdown!(s)
```

# Writing an autonomous vehicle client

The file example_project.jl outlines a recommended architecture for ingesting sensor messages and creating vehicle commands.

# Some other tools
## To test A* path finding
```julia
julia> include("./src/tools/astar.jl")
5[13]
13[18, 20]
20[49]
49[51]
51[53]
53[55]
55[57]
57[59]
59[61]
61[73, 74]
74[75]
75[77]
77[79]
79[81]
81[83]
83[7, 8, 9]
7[102]
102[100]
```

