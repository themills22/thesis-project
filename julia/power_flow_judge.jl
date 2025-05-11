#import packages
using HomotopyContinuation, LinearAlgebra, DataStructures, Dates, Combinatorics, NPZ

#number of nodes
n = 4;

#define variables
@var x[1:n]
@var y[1:n]

#number of trials
its = 1000000

#define variables
@var x[1:n]
@var y[1:n]

#define variables and equations to be repeatedly solved
#define the edge variables, cheat to make it match an undirected graph edge matrix
b = Array{Variable}(undef, n, n)
for (i, j) in collect(combinations(1:n, 2))
    v = Variable(:b, i, j)
    b[i, j] = v
    b[j, i] = v
end


function create_susceptance(k, m)
    if m == 1
        return b[k, m] * y[k]
    end

    if m == k
        return 0
    end

    return b[k, m] * (x[m] * y[k] - x[k] * y[m])
end

power_equations = Array{Expression}(undef, n - 1)
for i in 1:n - 1
    k = i + 1
    p = [create_susceptance(k, m) for m in 1:n]
    power_equations[i] = sum(p)
end

constraint_equations = [x[i] ^ 2 + y[i] ^ 2 - 1 for i in 2:n]

b_flat = [b[i, j] for (i, j) in collect(combinations(1:n, 2))]
variables = vcat(view(x, 2:n), view(y, 2:n))
equations = vcat(power_equations, constraint_equations)

#define system of equations
F = System(equations; variables = variables, parameters = b_flat);

#solve system once with generic complex parameters
start_parameters = rand(Complex{Float64}, length(b_flat));
R = solve(F(variables, start_parameters), show_progress = false);
S = solutions(R);

base_directory = "data\\power-flow\\graph"
judge_directory = joinpath(base_directory, string(n), "judged")
guess_directory = joinpath(base_directory, string(n), "guesses")
guess_path = argmax(mtime, readdir(guess_directory, join=true))

npz = npzread(guess_path)
initial_counts = Vector{Int64}()
final_counts = Vector{Int64}()
initial_systems = npz["initial_systems"]
final_systems = npz["systems"]
guess_counts = npz["solution_counts"]
for i in axes(final_systems)[1]
    initial_parameters = initial_systems[i, :]
    R_initial = solve(F, S; start_parameters=start_parameters, target_parameters=initial_parameters, show_progress=false)
    push!(initial_counts, length(real_solutions(R_initial)))

    final_parameters = final_systems[i, :]
    R_final = solve(F, S; start_parameters=start_parameters, target_parameters=final_parameters, show_progress=false)
    push!(final_counts, length(real_solutions(R_final)))
end

judge_file = splitdir(guess_path)[2]
npzwrite(joinpath(judge_directory, judge_file), Dict("initial_counts" => initial_counts, "guess_counts" => guess_counts,
    "solution_counts" => final_counts, "initial_systems" => initial_systems, "systems" => final_systems))

# for file in readdir(guess_directory, join=true)
#     npz = npzread(file)
#     initial_counts = Vector{Int64}()
#     final_counts = Vector{Int64}()
#     initial_systems = npz["initial_systems"]
#     final_systems = npz["systems"]
#     guess_counts = npz["solution_counts"]
#     for i in axes(final_systems)[1]
#         initial_parameters = initial_systems[i, :]
#         R_initial = solve(F, S; start_parameters=start_parameters, target_parameters=initial_parameters, show_progress=false)
#         push!(initial_counts, length(real_solutions(R_initial)))

#         final_parameters = final_systems[i, :]
#         R_final = solve(F, S; start_parameters=start_parameters, target_parameters=final_parameters, show_progress=false)
#         push!(final_counts, length(real_solutions(R_final)))
#     end

#     npzwrite(joinpath(judge_directory, file), Dict("initial_counts" => initial_counts, "guess_counts" => guess_counts,
#         "solution_counts" => final_counts, "initial_systems" => initial_systems, "systems" => final_systems))
# end