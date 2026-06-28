module PowerFlow

# import packages
using HomotopyContinuation, LinearAlgebra, DataStructures, Dates, Combinatorics, NPZ, Graphs, SparseArrays, GraphIO.EdgeList, TickTock

global matrix_system_cache = Dict()

# here for convenience; relies on only one power flow system being used per usage of this module
global power_flow_system_cache = Dict()

function get_non_nothing(graph_dict, vertex1, vertex2)
    item = get(graph_dict, Edge(vertex1, vertex2), missing)
    if !isequal(item, missing)
        return item
    end
    return get(graph_dict, Edge(vertex2, vertex1), missing)
end

function to_undirected_graph(graph)
    undirected_graph = SimpleGraph(nv(graph))
    for edge in edges(graph)
        add_edge!(undirected_graph, src(edge), dst(edge))
    end
    return undirected_graph
end

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

function create_power_flow_system(graph_path)
    item = get(power_flow_system_cache, graph_path, missing);
    if !isequal(item, missing)
        return item;
    end
    
    G = loadgraph(graph_path, EdgeListFormat());
    G = to_undirected_graph(G);

    #dimension of ellipses
    n = 2*nv(G);

    #define variables
    @var x[1:n]

    parameters = Dict(e => Variable(:b, src(e), dst(e)) for e in edges(G))
    susceptance_equations = Expression[]
    for v in 2:nv(G)
        rows = Int[]
        cols = Int[]
        vals = Expression[]
        for u in neighbors(G, v)
            coeff = get_non_nothing(parameters, v, u) / 2

            # top right block
            push!(rows, v)
            push!(cols, u + nv(G))
            push!(vals, -coeff)

            push!(rows, u)
            push!(cols, v + nv(G))
            push!(vals, coeff)

            # bottom left block
            push!(rows, v + nv(G))
            push!(cols, u)
            push!(vals, coeff)

            push!(rows, u + nv(G))
            push!(cols, v)
            push!(vals, -coeff)
        end

        matrix = sparse(rows, cols, vals, n, n)
        push!(susceptance_equations, x' * matrix * x - 1)
    end

    slack_equations = [-1 + x[1] ^ 2, x[1 + nv(G)]]
    one_equations = [x[i]^2 + x[i + nv(G)]^2 - 1 for i in 2:nv(G)]
    Eqs = vcat(susceptance_equations, slack_equations, one_equations)

    #reshape parameters into vector
    edge_list = collect(edges(G))
    edge_list = sort(edge_list, by = e -> (src(e), dst(e)))
    p = [parameters[e] for e in edge_list];

    #define system of equations
    F = System(Eqs; variables = x, parameters = p);

    # solve system once with generic complex parameters
    p0 = rand(Complex{Float64}, length(p));
    R = solve(F(x, p0), show_progress = false);
    S = solutions(R);
    power_flow_system_cache[graph_path] = (F, S, p0);
    return (F, S, p0);
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
        message = sprint(showerror, e, catch_backtrace());
        println(message);
        return Nothing;
    end
end

# sorted_values are expected to be sorted by the edge tuples
function judge_power_flow_systems(graph_path, sorted_values)
    # number of nodes
    F, start_sols, start_param = create_power_flow_system(graph_path)
    try
        counts = [];
        for i in axes(sorted_values, 1)
            system = sorted_values[i, :]

            # solve system
            R1 = solve(F, start_sols; start_parameters=start_param, target_parameters=system, show_progress =false);
            append!(counts, length(real_solutions(R1)));
        end

        return counts;
    catch e
        message = sprint(showerror, e, catch_backtrace());
        println(message);
        return Nothing;
    end
end

#This function inputs a non-real root (root) which you want to make real, the polynomial system of equations (F),
#the parameters for which F(root) = 0 (p0) and the number of real roots to F(x;p0)=0 (nReals)
#This function outputs the new parameter values, new root, norm of imaginary part and counter value
#If it terminates before counter = 10, then the new root will be real, the norm of the imaginary part of the new
#root will be <0.01 and the new parameter values p0 will have more real roots then nReals
function random_hill_climb(root, F, p0, nReals)
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

function hill_climb(F, start_sols, p0, iteration_cap)

    NRealSols = [];
    times = []

    #collect real and nonreal solutions
    realSols = real_solutions(start_sols);
    nonreal = findall(i -> norm(imag.(sols[i])) > 1e-8, eachindex(sols));


    #identify nonreal root you wish to bring together
    rootidx = nonreal[1];
    root = sols[rootidx]

    #run one iteration of random hill climbing
    tick()
    p0, Newroot, newOrigNorm, ncounter = random_hill_climb(root, F, p0, length(realSols))

    # do I want this?
    # p0 = rand(length(p));
    for its in 1:iteration_cap
        println(its)

        R = solve(F(x, p0));
        sols = solutions(R);

        #collect real and nonreal solutions
        realSols = real_solutions(R);
        append!(NRealSols, length(realSols));
        push!(times, peektimer())

        if length(realSols)==length(sols)
            break
        end

        nonreal = findall(i -> norm(imag.(sols[i])) > 1e-8, eachindex(sols));


        #identify nonreal root you wish to bring together
        rootidx = nonreal[1];
        root = sols[rootidx];

        #run one iteration of random hill climbing
        p0, Newroot, newOrigNorm, ncounter = random_hill_climb(root, F, p0, length(realSols));
    end

    tock()
    return NRealSols, times()
end

function hill_climb_matrix_systems(start_systems, iteration_cap)
    # number of nodes
    n = size(start_systems, 2);
    F, start_sols, start_param = create_matrix_system(n);
    try
        all_solutions = [];
        all_times = [];
        for i in axes(start_systems, 1)
            start_system = start_systems[i, :, :, :]
            new_params = [start_system[j, :, :]*start_system[j, :, :]' for j in 1:n];
            new_params = collect(Iterators.flatten(new_params));

            solutions, times = hill_climb(F, start_sols, new_params, iteration_cap);
            append!(all_solutions, solutions);
            append!(all_times, times);
        end

        return all_solutions, all_times;
    catch e
        message = sprint(showerror, e, catch_backtrace());
        println(message);
        return Nothing;
    end
end

# sorted_values are expected to be sorted by the edge tuples
function hill_climb_flow_systems(graph_path, sorted_values, iteration_cap)
    # number of nodes
    F, start_sols, start_param = create_power_flow_system(graph_path)
    try
        all_solutions = [];
        all_times = [];
        for i in axes(sorted_values, 1)
            system = sorted_values[i, :]

            solutions, times = hill_climb(F, start_sols, system, iteration_cap);
            append!(all_solutions, solutions);
            append!(all_times, times);
        end

        return all_solutions, all_times;
    catch e
        message = sprint(showerror, e, catch_backtrace());
        println(message);
        return Nothing;
    end
end

end