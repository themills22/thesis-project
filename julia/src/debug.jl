include(joinpath(@__DIR__, "PowerFlow.jl"))
import .PowerFlow: hill_climb_power_flow_systems, judge_power_flow_systems, hill_climb_matrix_systems

using NPZ

# values = rand(5, 4, 4, 4) .- 0.5
# counts, times = hill_climb_matrix_systems(values, 3)
# println(counts)

graph_path = "D:\\deep-reinforcement-learning\\thesis-project-tertiary\\data\\graphs\\showtime\\case-4gs.edgelist"

values = (rand(20, 4) .- 0.5) * 2
# counts = judge_power_flow_systems(graph_path, values)
# println(counts)

# initial_systems = npzread("D:\\deep-reinforcement-learning\\thesis-project-tertiary\\data\\points\\initial-case4gs.npy")
# # subset = initial_systems[1:2, :]
solutions, counts = hill_climb_power_flow_systems(graph_path, values, 3)
# npzwrite("D:\\deep-reinforcement-learning\\thesis-project-tertiary\\results\\compare\\solutions-case4gs.npz", Dict("solutions" => solutions, "counts" => counts))