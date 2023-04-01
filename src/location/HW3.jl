module HW3

using GLMakie
using Makie
using GeometryBasics
using LinearAlgebra
using Infiltrator
using StatsBase
using DelimitedFiles
#using Statistics

include("mapping.jl")
include("icp.jl")

export manual_collection, label_generated_map

end # module HW3
