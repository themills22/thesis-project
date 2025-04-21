#import packages
using HomotopyContinuation, LinearAlgebra, DataStructures, Dates, Combinatorics, NPZ

#number of nodes
n = 4;

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

reference_bus = [ x[1], y[1] ]
b_flat = [b[i, j] for (i, j) in collect(combinations(1:n, 2))]
parameters = vcat(reference_bus, b_flat)
power_equations = Array{Expression}(undef, n - 1)
for i in 1:n - 1
    k = i + 1
    p = [k != m ? b[k, m] * (x[m] * y[k] - x[k] * y[m]) : 0 for m in 1:n]
    power_equations[i] = sum(p)
end

constraint_equations = [x[i] ^ 2 + y[i] ^ 2 - 1 for i in 2:n]

variables = vcat(view(x, 2:n), view(y, 2:n))
equations = vcat(power_equations, constraint_equations)

#define system of equations
F = System(equations; variables = variables, parameters = parameters);

#solve system once with generic complex parameters
start_param = vcat([1, 0], rand(Complex{Float64}, length(b_flat)));
R = solve(F(variables, start_param), show_progress = false);
S = solutions(R);

#select solutions to start_param up to +/- symmetry
Gr = GroupActions(x -> (-1*x ));
m = multiplicities(S, group_action = Gr);
start_sols = [S[m[i][1]] for i in 1:length(m)];

directory = "data\\power-flow\\guesses\\4\\"
save_directory = "data\\power-flow\\judged\\4\\"

for file in readdir(directory)
    npz = npzread(joinpath(directory, file))
    actual_counts = []
    systems = npz["systems"]
    guess_counts = npz["solution_counts"]
    for i in axes(systems)[1]
        target_parameters = vcat([1, 0], systems[i, :])
        R1 = solve(F, start_sols; start_parameters=start_param, target_parameters=target_parameters, show_progress=false)
        S1 = solutions(R1)
        actual_count = 2 * length(real_solutions(R1))
        append!(actual_counts, actual_count)
    end

    npzwrite(joinpath(save_directory, file), Dict("systems" => systems, "solution_counts" => actual_counts, "guess_counts" => guess_counts))
end