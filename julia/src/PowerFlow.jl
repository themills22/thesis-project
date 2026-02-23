module PowerFlow

# import packages
using HomotopyContinuation, LinearAlgebra, DataStructures, Dates, Combinatorics, NPZ

function judge_graph_systems(systems)
    # number of nodes
    n = size(systems, 2);

    # define variables
    @var x[1:n]
    @var y[1:n]

    # define variables and equations to be repeatedly solved
    # define the edge variables, cheat to make it match an undirected graph edge matrix
    b = Array{Variable}(undef, n, n)
    for (i, j) in collect(combinations(1:n, 2))
        v = Variable(:b, i, j)
        b[i, j] = v
        b[j, i] = v
    end

    power_equations = Array{Expression}(undef, n - 1)
    for i in 1:n - 1
        k = i + 1
        p = [create_susceptance(b, x, y, k, m) for m in 1:n]
        power_equations[i] = sum(p)
    end

    constraint_equations = [x[i] ^ 2 + y[i] ^ 2 - 1 for i in 2:n]

    b_flat = [b[i, j] for (i, j) in collect(combinations(1:n, 2))]
    variables = vcat(view(x, 2:n), view(y, 2:n))
    equations = vcat(power_equations, constraint_equations)

    # define system of equations
    F = System(equations; variables = variables, parameters = b_flat);

    #solve system once with generic complex parameters
    start_parameters = rand(Complex{Float64}, length(b_flat));
    R = solve(F(variables, start_parameters), show_progress = false);
    S = solutions(R);

    index_chunks = Iterators.partition(axes(initial_systems)[1], length(initial_systems) ÷ Threads.nthreads())
    judge_tasks = map(index_chunks) do index_chunk
        Threads.@spawn judge(systems, index_chunk, start_parameters, F, S)
    end
    counts = fetch.(judge_tasks)
    return collect(Iterators.flatten(counts))
end

function create_susceptance(b, x, y, k, m)
    if m == 1
        return b[k, m] * y[k]
    end

    if m == k
        return 0
    end

    return b[k, m] * (x[m] * y[k] - x[k] * y[m])
end

function judge_partition(systems, partition, start_parameters, F, S)
    counts = Vector{Int64}()
    for i in partition
        target_parameters = systems[i, :]
        R = solve(F, S; start_parameters=start_parameters, target_parameters=target_parameters, show_progress=false)
        push!(counts, length(real_solutions(R)))
    end

    return counts
end

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