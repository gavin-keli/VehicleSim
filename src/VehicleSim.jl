module VehicleSim

using ColorTypes
using Dates
using GeometryBasics
using MeshCat
using MeshCatMechanisms
using Random
using Rotations
using RigidBodyDynamics
using Infiltrator
using LinearAlgebra
using SparseArrays
using Suppressor
using Sockets
using Serialization
using StaticArrays
using DifferentialEquations
using AStarSearch

include("view_car.jl")
include("objects.jl")
include("sim.jl")
include("client.jl")
include("control.jl")
include("sink.jl")
include("measurements.jl")
include("map.jl")
#include("project.jl")
include("project2.jl")
include("trajectory.jl")

export server, shutdown!, keyboard_client, example_client, project_client

end
