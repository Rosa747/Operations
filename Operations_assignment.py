
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

nodes_df = pd.read_excel('Node_operations.xlsx')


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
        {"name": "route_arr_1a", "A/D": "A", "route": route_arr_1a, "edges": edges(route_arr_1a, directed=True)},
        {"name": "route_arr_1b", "A/D": "A", "route": route_arr_1b, "edges": edges(route_arr_1b, directed=True)},
        {"name": "route_dep_1",  "A/D": "D", "route": route_dep_1,  "edges": edges(route_dep_1, directed=True)},
    ])

    return R

nodes = nodes_df["name"].tolist()

def routes_with_edge(edge):
    return R.loc[R["edges"].apply(lambda edge_list: edge in edge_list), "name"].tolist()

def find_separation(leading_ac, trailing_ac):
    return Sep.loc[P.loc[P['id'] == trailing_ac, 'WTC'], P.loc[P['id'] == leading_ac, 'WTC']]

def length_edge(edge):
    u, v = edge
    delta_x = nodes_df.loc[nodes_df['name'] == v, 'x'].values[0] - nodes_df.loc[nodes_df['name'] == u, 'x'].values[0]
    delta_y = nodes_df.loc[nodes_df['name'] == v, 'y'].values[0] - nodes_df.loc[nodes_df['name'] == u, 'y'].values[0]
    length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    return length

# TODO redefine route1 and route2 to set of all possible routes for ac i and ac j
# def get_common_nodes(route1, route2):
#     return list(set(route1) & set(route2))


R = create_routes()

print(routes_with_edge(("a1", "p2")))


n_aircraft = 2  # or len(aircraft_list) if you load data
n_nodes = 42 # number of nodes in the network (0,1,2,3) arrival (4,5,6,7,8) departure (9...) taxiway nodes
n_routes = 3  # number of possible routes

M = 1e4
Suv_max = 30 * 0.514444 #max speed in m/s
e_l = 3 #edge capacity for all runway exits
Tidep = 55 #departure time interval in seconds for all D

# Index set for aircraft
# Set of aircraft
P = pd.DataFrame([{ "id": "AC1", "A/D": "A", "ETD": pd.Timestamp("2024-06-01 10:00:00"), "WTC": "large"},
                  { "id": "AC2", "A/D": "A", "ETD": pd.Timestamp("2024-06-01 10:00:00"), "WTC": "large"}, 
                    { "id": "AC3", "A/D": "D", "PBT": pd.Timestamp("2024-06-01 10:00:00"), "WTC": "large"}])


# TODO we need N where N_i^p is the pth route for aircraft i, where that then is a list of nodes in sequence of the route
for aircraft in P:
    if aircraft["A/D"] == "A":
        P[aircraft]["routes"] = [R[i]["route"] for i in R.loc[ R["A/D"] == "A" ]]

    elif aircraft["A/D"] == "D":
        P[aircraft]["routes"] = [R[i]["route"] for i in R.loc[ R["A/D"] == "D" ]]
    

# Create subsets of aircraft
A = P.loc[ P["A/D"] == "A"]["id"].to_list()
D = P.loc[ P["A/D"] == "D"]["id"].to_list()
P_list = P["id"].to_list()
print(A)

# Create subsets of nodes
a = ["l3", "l4", "l5", "l6"]  # left side arrival runway exits
b = ["p2", "p3", "p4", "p5"]  # right side arrival runway exits
c = ["a1", "a2", "a3", "a4"]  # departure runway entries

# TODO subset of edges: L = all exit taxi edges


# # Set of nodes in each route #TODO hoe definieren we welke AC welke route mag nemen?
# for i in P:
#     Ypsilon = {r: R["route"][r] for r in range(n_routes)}
# # Set of edges in each route #TODO hoe definieren we welke AC welke route mag nemen?
# for i in P:
#     Lambda = {r: [(R["route"][r][k], R["route"][r][k+1]) for k in range(len(R["route"][r])-1)] for r in range(n_routes)}


# #TODO:
# # edge_routes is list of route IDs that include edge u,v
# edge_routes = {
# (u,v): [r for r in R if any(R_nodes[r][k]==u and R_nodes[r][k+1]==v for k in range(len(R_nodes[r])-1))]
# for (u,v) in E}
    
V = pd.DataFrame({"type": ["small", "large", "heavy", "B757"], # separation minima between aircraft types in seconds
                      "small": [59, 59, 59, 59], 
                      "large": [88, 61, 61, 61], 
                      "heavy": [109, 109, 90, 109], 
                      "B757": [110, 91, 91, 91]})

Sep = pd.DataFrame({"type": ["small", "large", "heavy", "B757"], 
                      "small": [40, 45, 55, 60], 
                      "large": [45, 50, 60, 65], 
                      "heavy": [55, 60, 70, 75], 
                      "B757": [60, 65, 75, 80]})
print(V)
# Create a simple model
model = gp.Model("test")

# List with tuples of aircraft ID combinations
a_k_combinations = [(A[i], A[j], nodes[k]) for i in range(len(A)) for j in range(len(A)) if i != j if A[i] < A[j] for k in range(len(nodes))]


# Decision variables
Z = model.addVars(a_k_combinations, name = "Z", vtype=GRB.BINARY)


t = model.addVars([(P_list[i], nodes[k]) for i in range(len(A)) for k in range(len(nodes))], vtype=GRB.CONTINUOUS, lb=0, name="t")


rho = model.addVars([(P_list[i], P_list[j]) for i in range(len(A)) for j in range(len(P_list)) if i<j], vtype=GRB.BINARY, name="rho")

#print("For gamma: ", [(P_list[i], R.loc[j, "name"]) for i in range(len(A)) for j in range(len(R))])
Gamma = model.addVars([(P_list[i], R.loc[j, "name"]) for i in range(len(A)) for j in range(len(R))], vtype=GRB.BINARY, name="Gamma")


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
    (gp.quicksum(Gamma[i,r] for r in routes_with_edge((u,v))) +
    gp.quicksum(Gamma[j,r] for r in routes_with_edge((u,v))))
    for i in A for j in A if i != j for (u,v) in E),
name="no_overtake_upper")

# Constraint 12: 
model.addConstrs(
(Z[i,j,u] - Z[i,j,v] >= (
    gp.quicksum(Gamma[i,r] for r in routes_with_edge((u,v))) +
    gp.quicksum(Gamma[j,r] for r in routes_with_edge((u,v)))
) - 2
    for i in A for j in A if i != j for (u,v) in E),
name="no_overtake_lower")

# Constraint 13:  
model.addConstrs(
    (z[i,j,u] - z[i,j,v] <= 2 -
    (gp.quicksum(Gamma[i,r] for r in routes_with_edge((u,v))) +
    gp.quicksum(Gamma[j,r] for r in routes_with_edge((u,v))))
    for i in A for j in A if i != j for (u,v) in E),
name="headon_upper")

# Constraint 14:  
model.addConstrs(
(Z[i,j,u] + Z[i,j,v] >= (
    gp.quicksum(Gamma[i,r] for r in routes_with_edge((u,v))) +
    gp.quicksum(Gamma[j,r] for r in routes_with_edge((u,v)))
) - 2
    for i in A for j in A if i != j for (u,v) in E),
name="headon_lower")

# Constraint 15: TODO UPDATE with dataframe
arrival_node = {i: routes[i][0] for i in routes}       # first node
model.addConstrs(
(t[j, routes[j][0]] >= ETD[j] for j in A_arrival),
name="arrival_time_window")

# Constraint 16: UPDATE with dataframe
departure_node = {i: routes[i][-1] for i in routes}    # last node
model.addConstrs(
(t[i, routes[i][-1]] >= PBT[i] for i in A_departure),
name="departure_time_window")

# Constraint 17,18 are not linearized: constraint 19,20 are linearized version
# Constraint 19: max taxi speed
model.addConstrs(
    (t[i, v] - t[i, u] >= routes_with_edge((u, v)) / Suv_max
     - M * (1 - gp.quicksum(Gamma[i, r] for r in routes_with_edge((u,v))))
     for i in A for (u, v) in E),
    name="taxi_speed_lower"
)

# Constraint 20: chosen not to be constrained by min taxi speed

# Constraint 21,22 are non-linear

# Constraint 23  ---- Need to update with N
# model.addConstrs(
#     (t[j, u] - t[i, u] - (t[i,v] - t[i, u]) * find_separation(i,j) / length_edge((u, v)) >=
#             - (3 - ((Z[i,j,u]) + gp.quicksum(Gamma[i, r] for r in routes_with_edge((u,v)))
#                    + gp.quicksum(Gamma[j, r] for r in R if u in 
#                    R_nodes[r]   # Here need N[r] wich is the nodes in sequence for route r
#                    ))) * M
#                     for i in A for j in A if i != j for (u, v) in E),
#                      name="sep_situation1")

# Constraint 24 ---- Need to update with N
model.addConstrs(
    (t[i,v] - t[j,v] - (t[j,v] - t[j, w]) * find_separation(i,j) / length_edge((w, v)) >=
            - (3 - ((Z[j,i,v]) + gp.quicksum(Gamma[j, r] for r in routes_with_edge((w,v)))
                   + gp.quicksum(Gamma[i, r] for r in R if v in 
                    R_nodes[r]) # Here need N[r] wich is the nodes in sequence for route r
                                 )) * M
                    for i in A for j in A if i != j for (w, v) in E),
                     name="sep_situation2")

# Constraint 28
d = ... # this should be the runway node
model.addConstrs((
    t[j,d] - t[i,d] - V[i,j] >= - (1- rho[i,j]) * M 
    for i in D for j in D if i!=j), name = 'runway_occupancy' )

# Constraint 31
model.addConstrs((
    t[j,b_k] - t[i,'17ra'] - Tidep >= - M * (1- rho[i,j])
    for i in D for j in A for b_k in b), name = 'runway_crossing_arrival' )

# Constraint 32
model.addConstrs((
    t[i,'17ra'] - t[j,a_k] - Tidep >= - M * (1- rho[i,j])
    for i in D for j in A for a_k in a), name = 'runway_crossing_departure' )

# Constraint 33
model.addConstrs((
    t[i,c_k] <= 
    for i in A for j in A for c_k in c), name = 'runway_crossing_departure' )

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
