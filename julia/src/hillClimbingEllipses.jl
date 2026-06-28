#import packages
using HomotopyContinuation, LinearAlgebra, DataStructures, Graphs, SparseArrays

## Generate ellipses and polynomial equations with given sparsity pattern

#Define graph being considered and adjacency matrix
G = path_graph(3);
add_edge!(G, 1, 3);
Ag = adjacency_matrix(G);

#Use adjacency matrix to get sparsity structure of ellipses
MSparse = [Diagonal(ones(nv(G))) Ag ; Ag Diagonal(ones(nv(G)))];
rows, cols, vals = findnz(MSparse);

#dimension of ellipses
n = 2*nv(G);

@var p[1:ne(G)]

#define variables
@var x[1:n]

#define parameters of matrices used to define ellipses
#@var A[1:n, 1:n, 1:n]
@var a[1:length(rows), 1:n]


#reshape parameters into vector of n nxn matrices with given sparsity
A_params = [sparse(rows, cols, a[1:end,i]) for i in 1:n];


#reshape parameters into vector
p = collect(Iterators.flatten(a));

#define parametric equations to be repeatedly solved
Eqs = [x'*A_params[i]*x - 1 for  i in 1:n];

#define system of equations
F = System(Eqs; variables = x, parameters = p);



### Random hill climbing

#Choose complex root to bring together
# rootidx = nonreal[1];
# root = sols[rootidx];
# origNorm = norm(imag.(root));


#This function inputs a non-real root (root) which you want to make real, the polynomial system of equations (F),
#the parameters for which F(root) = 0 (p0) and the number of real roots to F(x;p0)=0 (nReals)
#This function outputs the new parameter values, new root, norm of imaginary part and counter value
#If it terminates before counter = 10, then the new root will be real, the norm of the imaginary part of the new
#root will be <0.01 and the new parameter values p0 will have more real roots then nReals
function randomHillClimb(root, F, p0, nReals)
counter = 0
origNorm = norm(imag.(root));
  while origNorm > 0.01
    #generate 10 random directions
    randDirections = [0.01*randn(length(p0)) for i in 1:10];

    #run Newtons method on 10 random directions to see change in root
    newRoots = [solution(newton(F, root, p0+randDirections[i])) for i in 1:length(randDirections)];
    newNorms = [norm(imag.(newRoots[i])) for i in 1:length(newRoots)];

    #find all directions, norms of imaginary part and roots where complex part of root gets smaller
    potentialDirectionsidx = findall(x->x<0, newNorms-origNorm*ones(10));
    potDirs = randDirections[potentialDirectionsidx];
    potNewNorms = newNorms[potentialDirectionsidx];
    potNewRoots = newRoots[potentialDirectionsidx];

    if length(potentialDirectionsidx)==0
      continue
    end

    potDirs2 = [];
    potNewNorms2 = [];
    potNewRoots2 = [];

    #make sure new potential direction does not make other real solutions come together
    for i in 1:length(potentialDirectionsidx)
      R1 = solve(F(x, p0 + potDirs[i]));
      realSols1 = real_solutions(R1);
      if nReals==0 && length(realSols1) == 0
        append!(potDirs2, [potDirs[i]])
        append!(potNewNorms2, potNewNorms[i])
        append!(potNewRoots2, [potNewRoots[i]])
      elseif nReals > length(realSols1)
        continue
      else
        dists1 = [norm(realSols1[i] - realSols1[j]) for i in 1:length(realSols1) for j in i+1:length(realSols1)];

        if minimum(dists1) > 0.1
          append!(potDirs2, [potDirs[i]])
          append!(potNewNorms2, potNewNorms[i])
          append!(potNewRoots2, [potNewRoots[i]])
        end
      end
    end


    if length(potDirs2)>0
      counter = 0;
      minNorm, minidx = findmin(potNewNorms2);
      root = potNewRoots2[minidx];
      origNorm = minNorm
      p0 = p0+ potDirs2[minidx];
    else
      counter = counter + 1
    end

    if counter > 10
      break
    end

  end

  return p0, root, origNorm, counter

end

## Solve polynomial system for one choice of parameters

#solve system once with generic parameters p0
p0= rand(length(p));
R = solve(F(x, p0));
sols = solutions(R);

#collect real and nonreal solutions
realSols = real_solutions(R);
nonreal = findall(i -> norm(imag.(sols[i])) > 1e-8, eachindex(sols));


#identify nonreal root you wish to bring together
rootidx = nonreal[1];
root = sols[rootidx]

#run one iteration of random hill climbing
p0, Newroot, newOrigNorm, ncounter = randomHillClimb(root, F, p0, length(realSols))


global p0= rand(length(p));
NRealSols = [];
for its in 1:25
  println(its)

  R = solve(F(x, p0));
  sols = solutions(R);

  #collect real and nonreal solutions
  realSols = real_solutions(R);
  append!(NRealSols, length(realSols))

  println("Real vs total: ", length(realSols), " vs ", length(sols))
  if length(realSols)==length(sols)
    break
  end

  nonreal = findall(i -> norm(imag.(sols[i])) > 1e-8, eachindex(sols));


  #identify nonreal root you wish to bring together
  rootidx = nonreal[1];
  root = sols[rootidx]

  #run one iteration of random hill climbing
  global p0, Newroot, newOrigNorm, ncounter = randomHillClimb(root, F, p0, length(realSols))
end