module PowerFlow

# import packages
using HomotopyContinuation, LinearAlgebra, DataStructures, Dates, Combinatorics, NPZ

global matrix_system_cache = Dict()

function create_matrix_system(n)
    item = get(matrix_system_cache, n, missing);
    if !isequal(item, missing)
        return item;
    end

    # define variables
    @var x[1:n]

    # define parameters of matrices used to define ellipses
    @var A[1:n, 1:n, 1:n]

    # reshape parameters into vector of n nxn matrices
    A_params = [A[:, :, i] for i in 1:n];

    # reshape parameters into (n-1)^3 size vector
    A_flat = collect(Iterators.flatten(A_params));

    # define parametric equations to be repeatedly solved
    Eqs = [x'*A_params[i]*x - 1 for  i in 1:n];

    # define system of equations
    F = System(Eqs; variables = x, parameters = A_flat);

    # solve system once with generic complex parameters
    start_param = rand(Complex{Float64}, length(A_flat));
    R = solve(F(x, start_param), show_progress = false);
    S = solutions(R);

    # select solutions to start_param up to +/- symmetry
    Gr = GroupActions(x -> (-1*x ));
    m = multiplicities(S, group_action = Gr);
    start_sols = [S[m[i][1]] for i in 1:length(m)];
    matrix_system_cache[n] = (F, start_sols, start_param);
    return (F, start_sols, start_param);
end

function judge_matrix_systems(systems)
    # number of nodes
    n = size(systems, 2);
    F, start_sols, start_param = create_matrix_system(n);
    try
        counts = [];
        for i in axes(systems, 1)
            system = systems[i, :, :, :]
            new_params = [system[j, :, :]*system[j, :, :]' for j in 1:n];
            new_params = collect(Iterators.flatten(new_params));

            # solve system
            R1 = solve(F, start_sols; start_parameters=start_param, target_parameters=new_params, show_progress =false);
            append!(counts, 2 * length(real_solutions(R1)));
        end

        return counts;
    catch e
        println("Error occurred: $e");
        return Nothing;
    end
end

end