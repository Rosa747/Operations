
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

nodes_df = pd.read_excel('Node_operations.xlsx')

print(nodes_df.head())

def create_routes():

    route_arr_1a = ["a1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "l9", "l8", "l7", "l6", "k6", "d5"]
    route_arr_1b = ["a1", "p2", "17rc", "l3", "k3", "d2"]

    route_dep_1 = ["d2", "k3", "l3", "l2", "l1", "17ra"]


    # Define routes departure ac
        # couldn't find, think just shortest route gate-> departure node
    # Set of routes

    R = pd.DataFrame([{"name": "route_arr_1a", "route" : route_arr_1a},
                            {"name": "route_arr_1b", "route" : route_arr_1b},
                            {"name": "route_dep_1", "route" : route_dep_1}])


    def edges(route, directed=True):
        pairs = list(zip(route, route[1:]))
        return pairs if directed else [set(p) for p in pairs]

    R = pd.DataFrame([
        {"name": "route_arr_1a", "route": route_arr_1a, "edges": edges(route_arr_1a, directed=True)},
        {"name": "route_arr_1b", "route": route_arr_1b, "edges": edges(route_arr_1b, directed=True)},
        {"name": "route_dep_1",  "route": route_dep_1,  "edges": edges(route_dep_1, directed=True)},
    ])

    return R

# TODO functie die lengths per edges geeft
# Because I use:  L[(u, v)]

R = create_routes()

n_aircraft = 2  # or len(aircraft_list) if you load data
n_nodes = 42 # number of nodes in the network (0,1,2,3) arrival (4,5,6,7,8) departure (9...) taxiway nodes
n_routes = 3  # number of possible routes

M = 1e4
Suv_max = 30 * 0.514444 #max speed in m/s

# Index set for aircraft
# Set of aircraft
P = pd.DataFrame([{ "id": "AC1", "A/D": "A", "ETD": pd.Timestamp("2024-06-01 10:00:00")}, 
                    { "id": "AC2", "A/D": "D", "PBT": pd.Timestamp("2024-06-01 10:00:00")}])
R = range(n_routes)


# Set of nodes in each route #TODO hoe definieren we welke AC welke route mag nemen?
for i in P:
    Ypsilon = {r: R["route"][r] for r in range(n_routes)}
# Set of edges in each route #TODO hoe definieren we welke AC welke route mag nemen?
for i in P:
    Lambda = {r: [(R["route"][r][k], R["route"][r][k+1]) for k in range(len(R["route"][r])-1)] for r in range(n_routes)}


#TODO:
# edge_routes is list of route IDs that include edge u,v
edge_routes = {
(u,v): [r for r in R if any(R_nodes[r][k]==u and R_nodes[r][k+1]==v for k in range(len(R_nodes[r])-1))]
for (u,v) in E}



# Create a simple model
model = gp.Model("test")

# Decision variables
Z = model.addVar(A, A, V, vtype=GRB.BINARY, name="Z")
t = model.addVars(A, V, vtype=GRB.CONTINUOUS, lb=0, name="t")
rho = model.addVar(A, A, vtype=GRB.BINARY, name="rho")
Gamma = model.addVar(A, R, vtype=GRB.BINARY, name="Gamma")

# Objective: time sum
model.setObjective(gp.quicksum(t[i,u] for i in A for u in U), GRB.MINIMIZE)

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
(Z[i,j,u] - Z[i,j,v] >= (
    gp.quicksum(Gamma[i,r] for r in edge_routes[(u,v)]) +
    gp.quicksum(Gamma[j,r] for r in edge_routes[(u,v)])
) - 2
    for i in A for j in A if i != j for (u,v) in E),
name="no_overtake_lower")

# Constraint 13:  
model.addConstrs(
    (z[i,j,u] - z[i,j,v] <= 2 -
    (gp.quicksum(Gamma[i,r] for r in edge_routes[(u,v)]) +
    gp.quicksum(Gamma[j,r] for r in edge_routes[(v,u)]))
    for i in A for j in A if i != j for (u,v) in E),
name="headon_upper")

# Constraint 14:  
model.addConstrs(
(Z[i,j,u] + Z[i,j,v] >= (
    gp.quicksum(Gamma[i,r] for r in edge_routes[(u,v)]) +
    gp.quicksum(Gamma[j,r] for r in edge_routes[(v,u)])
) - 2
    for i in A for j in A if i != j for (u,v) in E),
name="headon_lower")

# Constraint 15: UPDATE with dataframe
arrival_node = {i: routes[i][0] for i in routes}       # first node
model.addConstrs(
(t[j, routes[j][0]] >= ETD[j] for j in A_arrival),
name="arrival_time_window")

# Constraint 16: UPDATE with dataframe
departure_node = {i: routes[i][-1] for i in routes}    # last node
model.addConstrs(
(t[i, routes[i][-1]] >= PBT[i] for i in A_departure),
name="departure_time_window")

# Constraint 17:
model.addConstrs(
    (t[i, v] - t[i, u] >= L[(u, v)] / Suv_max
     - M * (1 - gp.quicksum(Gamma[i, r] for r in edge_routes[(u, v)]))
     for i in A for (u, v) in E),
    name="taxi_speed_lower"
)

# Constraint 18:
model.addConstrs(
    (t[i, v] - t[i, u] <= L[(u, v)] / Suv_min
     + M * (1 - gp.quicksum(Gamma[i, r] for r in edge_routes[(u, v)]))
     for i in A for (u, v) in E),
    name="taxi_speed_upper"
)

# Optimize
model.optimize()
model.update()
model.write("model.lp")

# Print result
print(f"Optimization status: {m.Status}")
print(f"x = {x.X}, y = {y.X}")
print(f"Objective value = {m.ObjVal}")

# except gp.GurobiError as e:
# print(f"Gurobi Error: {e}")
# except Exception as e:
# print(f"Error: {e}")
