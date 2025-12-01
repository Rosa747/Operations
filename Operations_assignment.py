import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

nodes_df = pd.read_excel('Node_operations.xlsx')

# Function definitions
def create_routes():

    route_arr_1a = ["a1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "l9", "l8", "l7", "l6", "k6", "d5"]
    route_arr_1b = ["a1", "p2", "17rc", "l3", "k3", "d2"]

    route_dep_1 = ["d2", "k3", "l3", "l2", "l1", "17ra"]


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

def build_E(P, R):
    E = {}
    for _, row in P.iterrows():
        ac = row["id"]
        routes = row["routes"]
        E[ac] = {}
        if routes is None:
            continue
        for r in routes:
            E[ac][r] = R.loc[R["name"] == r, "edges"].values[0]
    return E

nodes = nodes_df["name"].tolist()

def routes_with_edge(edge):
    return R.loc[R["edges"].apply(lambda edge_list: edge in edge_list), "name"].tolist()

def find_separation(leading_ac, trailing_ac):
    return Sep.loc[P.loc[P['id'] == trailing_ac, 'WTC'], P.loc[P['id'] == leading_ac, 'WTC']]

def find_etd(id):
    return P.loc[P["id"] == id, "ETD"]

def length_edge(edge):
    u, v = edge
    delta_x = nodes_df.loc[nodes_df['name'] == v, 'x'].values[0] - nodes_df.loc[nodes_df['name'] == u, 'x'].values[0]
    delta_y = nodes_df.loc[nodes_df['name'] == v, 'y'].values[0] - nodes_df.loc[nodes_df['name'] == u, 'y'].values[0]
    length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    return length

# Assume P has a column "routes" which is a list of route names
def fill_upsilon(P, R):
    # Create a dictionary mapping route name -> route nodes
    route_dict = dict(zip(R["name"], R["route"]))
    
    # For each row in P, build the set of unique nodes
    P["Upsilon"] = P["routes"].apply(
        lambda route_names: {node for r in route_names for node in route_dict[r]}
    )
    
    return P

def get_first_node(aircraft_id):
    route_name = P.loc[P["id"] == aircraft_id, "routes"].iloc[0][0]  # e.g. "route_arr_1a"
    first_node = R.loc[R["name"] == route_name, "route"].iloc[0][0]  # e.g. "a1"
    return first_node

def get_last_node(aircraft_id):
    route_name = P.loc[P["id"] == aircraft_id, "routes"].iloc[0][0] # e.g. "route_arr_1a"
    last_node = R.loc[R["name"] == route_name, "route"].iloc[0][-1] # e.g. "d5"
    return last_node

# TODO redefine route1 and route2 to set of all possible routes for ac i and ac j
# def get_common_nodes(route1, route2):
#     return list(set(route1) & set(route2))

# TODO 
    # Define E #Lynn -> klaar (constraint 23/24 updaten nog)
    # Define N #Lynn -> Jim want R_nodes
    # Define U #Lynn
    # Define ETD #Done
    # Define A_arrival #bestaat niet meer?
    # Define A_departure #bestaat niet meer?
    # Define Routes #Jim Done
    # Define PBT #Jim Done
    # Define R_nodes # Jim done

    #Check of deze TODOs nog nodig zijn
    # TODO subset of edges: L = all exit taxi edges

    # # Set of nodes in each route #TODO hoe definieren we welke AC welke route mag nemen?
    # for i in P:
    #     Ypsilon = {r: R["route"][r] for r in range(n_routes)}
    # # Set of edges in each route #TODO hoe definieren we welke AC welke route mag nemen?
    # for i in P:
    #     Lambda = {r: [(R["route"][r][k], R["route"][r][k+1]) for k in range(len(R["route"][r])-1)] for r in range(n_routes)}




# print(routes_with_edge(("a1", "p2")))

n_aircraft = 2  # or len(aircraft_list) if you load data
n_nodes = 42 # number of nodes in the network (0,1,2,3) arrival (4,5,6,7,8) departure (9...) taxiway nodes
n_routes = 3  # number of possible routes

M = 1e4
Suv_max = 30 * 0.514444         # max speed in m/s
e_l = 3                         # edge capacity for all runway exits
Tidep = 55                      # departure time interval in seconds for all D



# Create dataframes:
R = create_routes()     # Set of routes for aircraft i

# Set of aircraft
P = pd.DataFrame([{ "id": "AC1", "A/D": "A", "ETD": 100, "WTC": "large"}, #ETD in sec
                  { "id": "AC2", "A/D": "A", "ETD": 200, "WTC": "large"}, #ETD in sec
                    { "id": "AC3", "A/D": "D", "PBT": 150, "WTC": "large"}]) #PBT in sec


P["routes"] = None
P["Upsilon"] = None
routes_A = R.loc[R["A/D"] == "A", "name"].tolist()
routes_D = R.loc[R["A/D"] == "D", "name"].tolist()

for idx, aircraft in P.iterrows():
    if aircraft["A/D"] == "A":
        P.at[idx, "routes"] = routes_A
    else:
        P.at[idx, "routes"] = routes_D
P = fill_upsilon(P, R)

# Create subsets of aircraft
A = P.loc[ P["A/D"] == "A"]["id"].to_list()
D = P.loc[ P["A/D"] == "D"]["id"].to_list()
P_list = P["id"].to_list()

# List with tuples of aircraft ID combinations
a_k_combinations = [(A[i], A[j], nodes[k]) for i in range(len(A)) for j in range(len(A)) if i != j if A[i] < A[j] for k in range(len(nodes))]


# Create subset of edges
E = build_E(P, R)
L = [("p2","a1"), ("p3","a2"), ("p4","a3"), ("p5","a4")]

# Create subsets of nodes
a = ["l3", "l4", "l5", "l6"]  # left side arrival runway exits
b = ["p2", "p3", "p4", "p5"]  # right side arrival runway exits
c = ["a1", "a2", "a3", "a4"]  # departure runway entries


# Route -> list of edges
route_edges = {
    row["name"]: row["edges"]
    for _, row in R.iterrows()
}

# Global set of all edges (w, v) in the network
# E_all = sorted({edge for edges in route_edges.values() for edge in edges})

# Aircraft characteristics: separation minima
V = pd.DataFrame({"type": ["small", "large", "heavy", "B757"], # separation minima between aircraft types in seconds
                      "small": [59, 59, 59, 59], 
                      "large": [88, 61, 61, 61], 
                      "heavy": [109, 109, 90, 109], 
                      "B757": [110, 91, 91, 91]})

a = ["l3", "l4", "l5", "l6"]  # left side arrival runway exits
b = ["p2", "p3", "p4", "p5"]  # right side arrival runway exits
c = ["a1", "a2", "a3", "a4"]  # departure runway entries

# -- Separation minima --
Sep = pd.DataFrame({"type": ["small", "large", "heavy", "B757"], 
                      "small": [40, 45, 55, 60], 
                      "large": [45, 50, 60, 65], 
                      "heavy": [55, 60, 70, 75], 
                      "B757": [60, 65, 75, 80]})

# Create a simple model
model = gp.Model("test")

# 1) Route: set of nodes for quick membership
route_nodes = {
    row["name"]: set(row["route"])
    for _, row in R.iterrows()
}

# 2) Aircraft: list of route names
P_routes = {
    row["id"]: row["routes"]
    for _, row in P.iterrows()
}
 

# U is the set of nodes aircraft i and j can both visit
U = {}

for i in P_list:
    for j in P_list:
        if i != j:
            nodes_i = P.loc[P["id"] == i, "Upsilon"].iloc[0]
            nodes_j = P.loc[P["id"] == j, "Upsilon"].iloc[0]
            U[(i, j)] = nodes_i.intersection(nodes_j)



# Decision variables
Z = model.addVars(a_k_combinations, name = "Z", vtype=GRB.BINARY)

t_index = [(aircraft_id, node) for aircraft_id, nodes in zip(P["id"], P["Upsilon"]) for node in nodes]
t = model.addVars(t_index, name="t", vtype=GRB.CONTINUOUS, lb=0)

rho = model.addVars([(P_list[i], P_list[j]) for i in range(len(P_list)) for j in range(len(P_list)) if i<j], vtype=GRB.BINARY, name="rho")

#print("For gamma: ", [(P_list[i], R.loc[j, "name"]) for i in range(len(A)) for j in range(len(R))])
Gamma = model.addVars([(P_list[i], R.loc[j, "name"]) for i in range(len(P_list)) for j in range(len(R))], vtype=GRB.BINARY, name="Gamma")

# Objective: time sum
model.setObjective(gp.quicksum(t[i,get_last_node(i)] for i in P_list), GRB.MINIMIZE)

# --- Constraints ---

# Constraint 6: one route
model.addConstrs(
(gp.quicksum(Gamma[acft, route] for route in P.loc[P["id"] == acft, "route"].iloc[0]) == 1 for acft in P_list),
name="one_route_per_aircraft")

# Constraint 7:

model.addConstrs(
    (
        Z[i, j, u] <= gp.quicksum(
            Gamma[i, r]
            for r in P_routes[i]          # only routes allowed for aircraft i
            if u in route_nodes[r]        # and that contain node u
        )
        for i in A
        for j in A
        for u in U[(i, j)] # U is here u ∈ Υi ∩ Υj
    ),
    name="Z_limited_by_i_route",
)


# Constraint 8: 
model.addConstrs(
    (
        Z[i, j, u] <= gp.quicksum(
            Gamma[i, r]
            for r in P_routes[i]          # only routes allowed for aircraft i
            if u in route_nodes[r]        # and that contain node u
        )
        for i in A
        for j in A
        for u in U[(i, j)] #U is here u ∈ Υi ∩ Υj
    ),
    name="Z_limited_by_j_route",
)

# Constraint 9: sequencing 
model.addConstrs(
    (
        Z[i, j, u]
        + Z[j, i, u]
        <= 3
           - gp.quicksum(
                 Gamma[i, r]
                 for r in P_routes[i]      # only routes that i can use
                 if u in route_nodes[r]    # that pass through node u
             )
           - gp.quicksum(
                 Gamma[j, r]
                 for r in P_routes[j]      # only routes that j can use
                 if u in route_nodes[r]
             )
        for i in A
        for j in A
        if i != j
        for u in U[(i, j)] # U is here u ∈ Υi ∩ Υj
    ),
    name="sequence_consistency_upper",
)

# Constraint 10: sequencing
model.addConstrs(
    (
        Z[i, j, u] + Z[j, i, u]
        >= 2 * (
                gp.quicksum(
                    Gamma[i, r]
                    for r in P_routes[i]      # routes aircraft i may use
                    if u in route_nodes[r]    # that pass through node u
                )
              + gp.quicksum(
                    Gamma[j, r]
                    for r in P_routes[j]      # routes aircraft j may use
                    if u in route_nodes[r]
                )
            ) - 3
        for i in A
        for j in A
        if i != j
        for u in U[(i, j)] #U is here u ∈ Υi ∩ Υj
    ),
    name="sequence_consistency_lower",
)


# Constraint 11 & 12: Overtaking constraints
# Loop over all aircraft pairs
for i in A:
    for j in A:
        if i == j:
            continue

        # Loop over aircraft i's possible routes
        for r_i in E[i]:
            edges_i = E[i][r_i]

            # Loop over aircraft j's possible routes
            for r_j in E[j]:
                edges_j = set(E[j][r_j])  # convert to set for faster lookup

                # Loop over edges in aircraft i's route
                for (u, v) in edges_i:
                    # Only consider edges that are also in aircraft j's route
                    if (u, v) in edges_j:
                        # Upper bound constraint (no overtaking)
                        model.addConstr(
                            Z[i, j, u] - Z[i, j, v] <= 2 -
                            (
                                gp.quicksum(Gamma[i, r] for r in E[i] if (u, v) in E[i][r]) +
                                gp.quicksum(Gamma[j, r] for r in E[j] if (u, v) in E[j][r])
                            ),
                            name=f"no_overtake_upper_{i}_{j}_{u}_{v}"
                        )

                        # Lower bound constraint (no overtaking)
                        model.addConstr(
                            Z[i, j, u] - Z[i, j, v] >=
                            (
                                gp.quicksum(Gamma[i, r] for r in E[i] if (u, v) in E[i][r]) +
                                gp.quicksum(Gamma[j, r] for r in E[j] if (u, v) in E[j][r])
                            ) - 2,
                            name=f"no_overtake_lower_{i}_{j}_{u}_{v}"
                        )


# Constraint 13 & 14: Head-on constraints (upper and lower)
for i in A:
    for j in A:
        if i == j:
            continue

        # Loop over aircraft i's possible routes
        for r_i in E[i]:
            edges_i = E[i][r_i]

            # Loop over aircraft j's possible routes
            for r_j in E[j]:
                edges_j = set(E[j][r_j])  # convert to set for fast lookup

                # Loop over edges in aircraft i's route
                for (u, v) in edges_i:
                    # Check if aircraft j travels the opposite edge (v,u)
                    if (v, u) in edges_j:
                        # Head-on upper bound
                        model.addConstr(
                            Z[i, j, u] - Z[i, j, v] <= 2 -
                            (
                                gp.quicksum(Gamma[i, r] for r in E[i] if (u, v) in E[i][r]) +
                                gp.quicksum(Gamma[j, r] for r in E[j] if (v, u) in E[j][r])
                            ),
                            name=f"headon_upper_{i}_{j}_{u}_{v}"
                        )

                        # Head-on lower bound
                        model.addConstr(
                            Z[i, j, u] + Z[i, j, v] >=
                            (
                                gp.quicksum(Gamma[i, r] for r in E[i] if (u, v) in E[i][r]) +
                                gp.quicksum(Gamma[j, r] for r in E[j] if (v, u) in E[j][r])
                            ) - 2,
                            name=f"headon_lower_{i}_{j}_{u}_{v}"
                        )


# Constraint 15: TODO UPDATE with dataframe
first_route_name = P.at[idx, "routes"][0]  # e.g. "route_arr_1a"
first_node = R.loc[R["name"] == first_route_name, "route"].iloc[0][0]  # e.g. "a1"
model.addConstrs(
    (
        t[j, get_first_node(j)] >= P.loc[P["id"] == j, "ETD"].iloc[0]
        for j in A
    ),
    name="arrival_time_window",
)

# Constraint 16: UPDATE with dataframe
first_route_name = P.at[idx, "routes"][0]  # e.g. "route_arr_1a"
first_node = R.loc[R["name"] == first_route_name, "route"].iloc[0][0]  # e.g. "a1"
model.addConstrs(
    (
        t[i, get_first_node(i)] >= P.loc[P["id"] == j, "PBT"].iloc[0]
        for i in D
    ),
    name="departure_time_window",
)

# Constraint 17,18 are not linearized: constraint 19,20 are linearized version
# Constraint 19: max taxi speed
for i in A:
    for r_i in E[i]:              # aircraft i's possible routes
        for (u, v) in E[i][r_i]:  # edges in route r_i
            model.addConstr(
                t[i, v] - t[i, u] <= (length_edge((u, v)) / Suv_max) *
                (M -M *(gp.quicksum(Gamma[i, r] for r in routes_with_edge((u, v)))) +
                gp.quicksum(Gamma[i, r] for r in routes_with_edge((u, v)))) ,
                name=f"taxi_speed_lower_{i}_{u}_{v}"
            )

# Constraint 20: chosen not to be constrained by min taxi speed

# Constraint 21,22 are non-linear

# Constraint 23  ---- Need to update with N
model.addConstrs(
    (t[j, u] - t[i, u] - (t[i,v] - t[i, u]) * find_separation(i,j) / length_edge((u, v)) >=
            - (3 - ((Z[i,j,u]) + gp.quicksum(Gamma[i, r] for r in routes_with_edge((u,v)))
                   + gp.quicksum(Gamma[i, r] for r in P_routes[i] if v in route_nodes[r])
                )) * M
                    for i in A for j in A if i != j for (u, v) in E),
                     name="sep_situation1")

# Constraint 24 ---- Need to update with N
model.addConstrs(
    (t[i,v] - t[j,v] - (t[j,v] - t[j, w]) * find_separation(i,j) / length_edge((w, v)) >=
            - (3 - ((Z[j,i,v]) + gp.quicksum(Gamma[j, r] for r in routes_with_edge((w,v)))
                   + gp.quicksum(Gamma[i, r] for r in P_routes[i] if v in route_nodes[r])
                                 )) * M
                    for i in A 
                    for j in A if i != j 
                    for (w, v) in E),
                     name="sep_situation2")



# Constraint 28
d = ... # this should be the runway node
model.addConstrs((
    t[j,d] - t[i,d] - V[i,j] >= - (1- rho[i,j]) * M 
    for i in D 
    for j in D if i!=j), name = 'runway_occupancy' )

# Constraint 31
model.addConstrs((
    t[j,b_k] - t[i,'17ra'] - Tidep >= - M * (1- rho[i,j])
    for i in D 
    for j in A 
    for b_k in b), name = 'runway_crossing_arrival' )

# Constraint 32
model.addConstrs((
    t[i,'17ra'] - t[j,a_k] - Tidep >= - M * (1- rho[i,j])
    for i in D 
    for j in A 
    for a_k in a), name = 'runway_crossing_departure' )

# Constraint 33
for l in L:                             # loop over exit edges
    for i in A:
        for j in A:
            if i == j:
                continue
            for r_i in E[i]:
                if l in [edge for edge in E[i][r_i]]:       # i uses edge l
                    for r_j in E[j]:
                        if l in [edge for edge in E[j][r_j]]:  # j uses edge l
                            model.addConstr(
                                t[i, l[0]] <= ETD,
                                name=f"exit_capacity_{i}_{j}_{l}"
                            )

# --- Generating results ---
model.optimize()
model.update()
model.write("model.lp")
print(f"Optimization status: {m.Status}")
print(f"x = {x.X}, y = {y.X}")
print(f"Objective value = {m.ObjVal}")

# except gp.GurobiError as e:
# print(f"Gurobi Error: {e}")
# except Exception as e:
# print(f"Error: {e}")
