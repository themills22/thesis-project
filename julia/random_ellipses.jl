#import packages
using HomotopyContinuation, LinearAlgebra, DataStructures

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
R = solve(F(x, start_param));
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

#solve polynomial system its times
for i in 1:its
if i%500 == 0
  println("Iteration : ", i)
end
  #define set of n random ellipses
  new_params = [randn(n,n) for i in 1:n];
  new_params = [new_params[i]*new_params[i]' for i in 1:n];

  #normalize ellipses to have norm 1
  new_params = [new_params[i]/norm(new_params[i]) for i in 1:n]
  new_params = collect(Iterators.flatten(new_params));

  #solve system
  R1 = solve(F, start_sols; start_parameters=start_param, target_parameters=new_params, show_progress =false)
  S1 = solutions(R1)
  append!(reals, length(real_solutions(R1)))
  append!(sols, length(S1))
  if length(S1)<2^(n-1)
    append!(bad_params, [new_params])
  end
end

length(reals)
print(counter(2*reals), " average = ", sum(2*reals)/length(reals))
