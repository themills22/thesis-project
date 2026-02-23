#import packages
using HomotopyContinuation, LinearAlgebra, DataStructures, Dates, NPZ

# order of arguments: <seed> <number of iterations>

#dimension of ellipses
n = 10;

#number of trials
its = 10000

#define variables
@var x[1:n]

#define parameters of matrices used to define ellipses
@var A[1:n, 1:n, 1:n]

#reshape parameters into vector of n nxn matrices
A_params = [A[:, :, i] for i in 1:n];

#reshape parameters into (n-1)^3 size vector
A_flat = collect(Iterators.flatten(A_params));

#define parametric equations to be repeatedly solved
Eqs = [x'*A_params[i]*x - 1 for  i in 1:n];

#define system of equations
F = System(Eqs; variables = x, parameters = A_flat);

#solve system once with generic complex parameters
start_param = rand(Complex{Float64}, length(A_flat));
R = solve(F(x, start_param), show_progress = false);
S = solutions(R);

#select solutions to start_param up to +/- symmetry
Gr = GroupActions(x -> (-1*x ));
m = multiplicities(S, group_action = Gr);
start_sols = [S[m[i][1]] for i in 1:length(m)];


#record # of real solutions in reals, # of solutions in sols and if there are any instances
reals = Vector{Int64}();
all_params = Array{Float64, 4}(undef, its, n, n, n);

#solve polynomial system its times
for i in 1:its

  #define set of n random ellipses
  new_params = randn(n, n, n);
  all_params[i, :, :, :] = new_params
  new_params = [new_params[i, :, :] * new_params[i, :, :]' for i in 1:n];
  new_params = collect(Iterators.flatten(new_params));

  #solve system
  R1 = solve(F, start_sols; start_parameters=start_param, target_parameters=new_params, show_progress =false)
  append!(reals, 2 * length(real_solutions(R1)))

  # show(R1)
  # show(length(real_solutions(R1)))

  # if i % 10 == 0
  #   println("Iteration: ", i)

  #   local file_name = "data\\power-flow\\ellipse\\" * string(n) * "\\" * Dates.format(now(UTC), "yyyy-mm-dd-HH-MM-SS-sss") * ".npz"
  #   npzwrite(file_name, Dict("systems" => all_params, "solution_counts" => reals))
  #   empty!(reals)
  # end
end