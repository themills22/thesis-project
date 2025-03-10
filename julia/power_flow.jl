#import packages
using HomotopyContinuation, LinearAlgebra, DataStructures, Dates, Combinatorics

#number of nodes
n = 4;

#number of trials
its = 100000

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

b_flat = [b[i, j] for (i, j) in collect(combinations(1:n, 2))]
power_equations = Array{Expression}(undef, n)
power_equations[1] = x[1] ^ 2 - 1
for i in 2:n
  p = [b[i][j] * (x[j] * y[i] - x[i] * y[j]) for j in 1:n]
  power_equations[i] = sum(p)
end

constraint_equations = [x[i] ^ 2 + y[i] ^ 2 - 1 for i in 1:n]

variables = vcat(x, y)
equations = vcat(power_equations, constraint_equations)

#define system of equations
F = System(equations; variables = variables, parameters = b_flat);

#solve system once with generic complex parameters
start_param = rand(Complex{Float64}, length(b_flat));
R = solve(F(x, start_param), show_progress = false);
S = solutions(R);

#select solutions to start_param up to +/- symmetry
Gr = GroupActions(x -> (-1*x ));
m = multiplicities(S, group_action = Gr);
start_sols = [S[m[i][1]] for i in 1:length(m)];


#record # of real solutions in reals, # of solutions in sols and if there are any instances
#where not all solutions are found, save parameters in bad_params
reals=[];
bad_params = [];
sols=[];
all_params = []

#solve polynomial system its times
for i in 1:its

  #define set of n random ellipses
  new_params = [randn(n,n) for (i, j) in collect(combinations(1:n, 2))]

  #normalize ellipses to have norm 1
  append!(all_params, new_params)

  #solve system
  R1 = solve(F, start_sols; start_parameters=start_param, target_parameters=new_params, show_progress =false)
  S1 = solutions(R1)
  append!(reals, length(real_solutions(R1)))
  append!(sols, length(S1))
  # if length(S1)<2^(n-1)
  #   append!(bad_params, [new_params])
  # end

  if i % 100 == 0
    println("Iteration : ", i)

    file_name = "matrices\\data\\" * Dates.format(now(UTC), "yyyy-mm-dd-HH-MM-SS-sss") * ".txt"
    open(file_name, "w") do file
      show(file, all_params)
      println(file)
      show(file, reals)
    end

    empty!(reals)
    empty!(bad_params)
    empty!(sols)
    empty!(all_params)
  end
end

if length(reals) > 0
  file_name = "matrices\\data\\" * Dates.format(now(UTC), "yyyy-mm-dd-HH-MM-SS-sss") * ".txt"
  open(file_name, "w") do file
    show(file, all_params)
    println(file)
    show(file, reals)
  end
end