#import packages
using HomotopyContinuation, LinearAlgebra, DataStructures, Dates, Combinatorics, NPZ

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


#record # of real solutions in reals, # of solutions in sols and if there are any instances
counts = Vector{Int64}()
all_parameters = Vector{Vector{Float64}}()

#solve polynomial system its times
for i in 1:its

  #define set of n random ellipses
  # target_parameters = [-1.0952639824628179, 0.03771609991708799, 1.0577394031194838, -2.286588504259049,  2.794930072093651, 2.092213677022314]

  #solve system
  target_parameters = randn(length(b_flat))
  R1 = solve(F, S; start_parameters=start_parameters, target_parameters=target_parameters, show_progress=false)
  S1 = solutions(R1)
  count = length(real_solutions(R1))

  push!(all_parameters, target_parameters)
  push!(counts, count)

  if i % 100000 == 0
    println("Iteration : ", i)

    file_name = joinpath("data\\power-flow\\graph\\", string(n), Dates.format(now(UTC), "yyyy-mm-dd-HH-MM-SS-sss") * ".npz")
    npzwrite(file_name, Dict("systems" => all_parameters, "solution_counts" => counts))
    empty!(counts)
    empty!(all_parameters)
  end
end