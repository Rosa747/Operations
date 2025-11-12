
import gurobipy as gp
from gurobipy import GRB

try:
    n_aircraft = 2  # or len(aircraft_list) if you load data
    n_nodes = 42 # number of nodes in the network (0,1,2,3) arrival (4,5,6,7,8) departure (9...) taxiway nodes
    n_routes = 3  # number of possible routes

    # TODO:
    # Define routes arrival ac
        # one shortest path, and another along 17R
    # Define routes departure ac
        # couldn't find, think just shortest route gate-> departure node
    # Set of routes
    R = [[]]

    # Index set for aircraft
    A = range(n_aircraft)
    V = range(n_nodes)
    R = range(n_routes)
    #TODO:
    # E = set of edges (u,v)
    # reverse_edge_routes[(v,u)] = list of routes using edge (v,u)
    # edge_routes[(u,v)] = list of routes using edge (u,v)
    # Gamma[i,r] = 1 if aircraft i uses route r
    # Z[i,j,u], Z[i,j,v] already defined

    # Create a simple model
    model = gp.Model("test")

    # Decision variables
    Z = model.addVar(A, A, V, vtype=GRB.BINARY, name="Z")
    t = model.addVars(A, V, vtype=GRB.CONTINUOUS, lb=0, name="t")
    rho = model.addVar(A, A, vtype=GRB.BINARY, name="rho")
    Gamma = model.addVar(A, R, vtype=GRB.BINARY, name="Gamma")

    # Objective: time sum
    model.setObjective(gp.quicksumt[i,u] for i in A for u in U), GRB.MINIMIZE)

    # Constraints:
    # Constraint 6: one route
    model.addConstrs(
    (gp.quicksum(Gamma[i, r] for r in R) == 1 for i in A),
    name="one_route_per_aircraft")

    # Constraint 7: 
    model.addConstrs(
    (Z[i, j, u] <= gp.quicksum(Gamma[i, r] for r in R if u in N[r])
     for i in A for j in A for u in U),
    name="Z_limited_by_i_route")

    # Constraint 8: 
    model.addConstrs(
    (Z[i, j, u] <= gp.quicksum(Gamma[j, r] for r in R if u in N[r])
     for i in A for j in A for u in U),
    name="Z_limited_by_j_route")

    # Constraint 9: sequencing 
    model.addConstrs(
    (Z[i,j,u] + Z[j,i,u] <= 3 -
     (gp.quicksum(Gamma[i,r] for r in R if u in N[r]) +
     gp.quicksum(Gamma[j,r] for r in R if u in N[r]))
     for i in A for j in A if i != j for u in U),
    name="sequence_consistency_upper")

    # Constraint 10: sequencing
    model.addConstrs(
    (Z[i,j,u] + Z[j,i,u] >= 
     2*(gp.quicksum(Gamma[i,r] for r in R if u in N[r]) +
        gp.quicksum(Gamma[j,r] for r in R if u in N[r])) - 3
     for i in A for j in A if i != j for u in U),
    name="sequence_consistency_lower")

    # Constraint 11: 
    model.addConstrs(
        (z[i,j,u] - z[i,j,v] <= 2 -
        (gp.quicksum(Gamma[i,r] for r in edge_routes[(u,v)]) +
        gp.quicksum(Gamma[j,r] for r in edge_routes[(u,v)]))
        for i in A for j in A if i != j for (u,v) in E),
    name="no_overtake_upper")

    # Constraint 12: 
    model.addConstrs(
    (Z[i,j,u] + Z[i,j,v] >= (
        gp.quicksum(Gamma[i,r] for r in edge_routes[(u,v)]) +
        gp.quicksum(Gamma[j,r] for r in edge_routes[(u,v)])
    ) - 2
     for i in A for j in A if i != j for (u,v) in E),
    name="no_overtake_lower")


    # Optimize
    m.optimize()

    # Print result
    print(f"Optimization status: {m.Status}")
    print(f"x = {x.X}, y = {y.X}")
    print(f"Objective value = {m.ObjVal}")

except gp.GurobiError as e:
    print(f"Gurobi Error: {e}")
except Exception as e:
    print(f"Error: {e}")
