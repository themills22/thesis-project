#import packages
using HomotopyContinuation, LinearAlgebra, DataStructures, Dates, Combinatorics

#number of nodes
n = 4;

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


#record # of real solutions in reals, # of solutions in sols and if there are any instances
reals=[];
sols=[];
all_params = []

#solve polynomial system its times
for i in 1:its

  #define set of n random ellipses
  new_params = randn(length(b_flat))

  #normalize ellipses to have norm 1
  append!(all_params, new_params)

  #solve system
  target_parameters = vcat([1, 0], new_params)
  R1 = solve(F, start_sols; start_parameters=start_param, target_parameters=target_parameters, show_progress =false)
  S1 = solutions(R1)
  append!(reals, 2 * length(real_solutions(R1)))
  append!(sols, length(S1))

  if i % 1000 == 0
    println("Iteration : ", i)

    file_name = "data\\power-flow\\graph\\" * n * "\\" * Dates.format(now(UTC), "yyyy-mm-dd-HH-MM-SS-sss") * ".npz"
    npzwrite(file_name, Dict("systems" => all_params, "solution_counts" => reals, "sols" => sols))
    empty!(reals)
    empty!(sols)
    empty!(all_params)
  end
end