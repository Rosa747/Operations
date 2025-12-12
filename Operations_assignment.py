import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

nodes_df = pd.read_excel('Node_operations.xlsx')

def build_model(params):

    # Function definitions
    def create_routes():

        route_arr_1a = ["a1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "l9", "l8", "l7", "l6", "k6", "d5"]
        route_arr_1b = ["a1", "p2", "17rc", "l3", "k3", "d2"]
        route_arr_2a = ["a3", "p4", "17re", "l5", "k5", "d4"]
        route_arr_2b = ["a3", "p4", "p5", "p6", "p7", "p8", "l9", "l8", "l7", "l6", "l5", "k5", "d4"]

        route_dep_1 = ["d1", "k2", "l2", "l1", "17ra"]
        route_dep_2 = ["d3", "k4", "l4", "l3", "l2", "l1", "17ra"]

        def edges(route, directed=True):
            pairs = list(zip(route, route[1:]))
            return pairs if directed else [set(p) for p in pairs]

        all_routes = [
            ("route_arr_1a", "A", route_arr_1a),
            ("route_arr_1b", "A", route_arr_1b),
            ("route_arr_2a", "A", route_arr_2a),
            ("route_arr_2b", "A", route_arr_2b),
            ("route_dep_1", "D", route_dep_1),
            ("route_dep_2", "D", route_dep_2),]

        R = pd.DataFrame([
            {"name": name,
            "A/D": AD,
            "route": route,
            "edges": edges(route, directed=True)}
            for (name, AD, route) in all_routes
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

    def find_separation(i, j, Sep_matrix):
        wtc_i = P.loc[P["id"] == i, "WTC"].item()
        wtc_j = P.loc[P["id"] == j, "WTC"].item()
        return Sep_matrix.loc[Sep_matrix["type"] == wtc_j, wtc_i].item()

    def find_vortex_separation(i, j, V_matrix):
        wtc_i = P.loc[P["id"] == i, "WTC"].item()
        wtc_j = P.loc[P["id"] == j, "WTC"].item()
        return V_matrix.loc[V_matrix["type"] == wtc_j, wtc_i].item()

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
            lambda route_names: {str(node) for r in route_names for node in route_dict[r]}
        )
        return P

    # def get_first_node(aircraft_id, P, R, Gamma):
    #     """
    #     Returns the first node of the chosen route for the aircraft.
    #     """
    #     # Find the chosen route for this aircraft
    #     chosen_route = None
    #     for r in P.loc[P["id"] == aircraft_id, "routes"].iloc[0]:
    #         if Gamma[aircraft_id, r].X > 0.5:
    #             chosen_route = r
    #             break

    #     if chosen_route is None:
    #         raise ValueError(f"No route chosen for aircraft {aircraft_id}")
        
    #      # Return the last node of that route
    #     return R.loc[R["name"] == chosen_route, "route"].iloc[0][0]

    def first_node(r_name):
        """Return the first node of route r_name"""
        return R.loc[R["name"] == r_name, "route"].iloc[0][0]
    
    def last_node(r_name):
        """Return the last node of route r_name"""
        return R.loc[R["name"] == r_name, "route"].iloc[0][-1]

    # def get_last_node(aircraft_id, P, R, Gamma):
    #     """
    #     Returns the last node of the chosen route for the aircraft.
    #     """
    #     # Find the chosen route
    #     chosen_route = None
    #     for r in P.loc[P["id"] == aircraft_id, "routes"].iloc[0]:
    #         if Gamma[aircraft_id, r].X > 0.5:
    #             chosen_route = r
    #             break

    #     if chosen_route is None:
    #         raise ValueError(f"No route chosen for aircraft {aircraft_id}")

    #     # Return the last node of that route
    #     return R.loc[R["name"] == chosen_route, "route"].iloc[0][-1]

    '''
    TODO redefine route1 and route2 to set of all possible routes for ac i and ac j
    def get_common_nodes(route1, route2):
        return list(set(route1) & set(route2))

        Ideetjes die het waard zijn om te onthouden:
        Lamda?_i = set of edges for aircraft i -> gebruiken wij E voor
        -> veel dubbele in voor als we veel routes hebben met gedeelde edges
        -> kan computationally heavy zijn

        Gekozen om de departure node voor alle aircraft hetzelfde te houden
        -> 17ra
        -> relevant voor constraint 28
    
        '''


    # --- Simulation Parameters ---
    # n_aircraft = 2  # or len(aircraft_list) if you load data
    # n_nodes = 42 # number of nodes in the network (0,1,2,3) arrival (4,5,6,7,8) departure (9...) taxiway nodes
    # n_routes = 3  # number of possible routes

    M = params.get("M", 10000)
    Suv_max = (30 * 0.514444) * params.get("taxi_speed_multiplier", 1.0)         # max speed in m/s
    Suv_min = 0.1                                                                # Arbitrary min taxi speed -> enforces non-zero positive
    # e_l = 3                                                                      # edge capacity for all runway exits
    Tidep = params.get("Tidep", 50)


    # Create dataframes:
    R = create_routes()     # Set of routes for aircraft i

    # --- Create Datasets ---

    # --- Aircraft ---
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
    A = P.loc[ P["A/D"] == "A" ]["id"].to_list()
    D = P.loc[ P["A/D"] == "D" ]["id"].to_list()
    P_list = P["id"].to_list()

    # List with tuples of aircraft ID combinations
    a_k_combinations = [(i, j, node) for i in P["id"] for j in P["id"] if i != j for node in nodes]
    
    # --- Node & Edge dataset ---
    E = build_E(P, R)
    L = [("p2","a1"), ("p3","a2"), ("p4","a3"), ("p5","a4")]
    a = ["l3", "l4", "l5", "l6"]  # left side departure runway in line with arrival runway exits
    b = ["p2", "p3", "p4", "p5"]  # right side departure runway in line with arrival runway exits
    c = ["a1", "a2", "a3", "a4"]  # arrival runway exits
    d = ["17ra"]                  # departure entry node (same all ac)

    # route_edges = {
    #     row["name"]: row["edges"]
    #     for _, row in R.iterrows()
    # }

    route_nodes = {
        row["name"]: set(row["route"])
        for _, row in R.iterrows()
    }

    # Separation minima
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
    
    Sep_scaled = Sep.copy()
    V_scaled   = V.copy()
    Sep_scaled.loc[:, Sep_scaled.columns != "type"] *= params.get("sep_multiplier", 1.0)
    V_scaled.loc[:,   V_scaled.columns != "type"]   *= params.get("vortex_multiplier", 1.0)

    U = {}
    for i in P_list:
        for j in P_list:
            if i != j:
                nodes_i = P.loc[P["id"] == i, "Upsilon"].iloc[0]
                nodes_j = P.loc[P["id"] == j, "Upsilon"].iloc[0]
                U[(i, j)] = nodes_i.intersection(nodes_j)
                

    P_routes = {
        row["id"]: row["routes"]
        for _, row in P.iterrows()
    }

    # Map route name → list of nodes
    route_map = dict(zip(R["name"], R["route"]))

    # Build N[i][p] = route list of p-th possible route for aircraft i
    N_dict = {
        i: {p: route_map[rname] for p, rname in enumerate(P_routes[i], start=1)}
        for i in P_list
    }
    N = pd.DataFrame.from_dict(N_dict, orient="index")
    N.index.name = "i"
    N.columns = [p for p in N.columns]

    # Create model
    model = gp.Model("test")

    # Decision variables
    Z =         model.addVars(a_k_combinations, name = "Z", vtype=GRB.BINARY)
    t_index =   [(aircraft_id, node) for aircraft_id, 
                 nodes_set in zip(P["id"], 
                 P["Upsilon"]) for node in nodes_set]
    t =         model.addVars(t_index, name="t", vtype=GRB.CONTINUOUS, lb=0)
    rho =       model.addVars([(acft_i, acft_j) 
                            for acft_i in P_list
                            for acft_j in P_list],
                            vtype=GRB.BINARY, name="rho")
    Gamma =     model.addVars([(P_list[i], R.loc[r, "name"]) 
                            for i in range(len(P_list)) 
                            for r in range(len(R))
                            if R.loc[r, "name"] in P_routes[P_list[i]]], 
                            vtype=GRB.BINARY, name="Gamma")




    # --- Objective Function ---
    model.setObjective(
        gp.quicksum( 
        gp.quicksum(Gamma[i, route] * (t[i,last_node(route)] - t[i, first_node(route)]) 
                    for route in P.loc[P["id"] == i, "routes"].iloc[0]
                                   )
                                   for i in P_list), GRB.MINIMIZE)

    # --- Constraints ---

    # Constraint 6: one route
    model.addConstrs(
        (gp.quicksum(Gamma[acft, route] 
                for route in P.loc[P["id"] == acft, "routes"].iloc[0]) == 1 
                for acft in P_list),
    name=f"one_route_for_aircraft",)

    # Constraint 7:
    model.addConstrs(
        (
            Z[i, j, u] <= gp.quicksum(
                Gamma[i, r]
                for r in P_routes[i]
                if u in route_nodes[r]
            )
            for i in P_list
            for j in P_list if i != j
            for u in U[(i, j)]                 # U is here u ∈ Υi ∩ Υj
        ),
        name="Z_limited_by_i_route",
    )

    # Constraint 8: 
    model.addConstrs(
        (
            Z[i, j, u] <= gp.quicksum(
                Gamma[j, r]
                for r in P_routes[j]
                if u in route_nodes[r]
            )
            for i in P_list
            for j in P_list if i != j
            for u in U[(i, j)]                 #U is here u ∈ Υi ∩ Υj
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
                    for r in P_routes[i]
                    if u in route_nodes[r] 
                )
            - gp.quicksum(
                    Gamma[j, r]
                    for r in P_routes[j]      # only routes that j can use
                    if u in route_nodes[r]
                )
            for i in P_list
            for j in P_list
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
                        for r in P_routes[i] 
                        if u in route_nodes[r] 
                    )
                + gp.quicksum(
                        Gamma[j, r]
                        for r in P_routes[j]
                        if u in route_nodes[r]
                    )
                ) - 3
            for i in P_list
            for j in P_list if i != j
            for u in U[(i, j)]  #U is here u ∈ Υi ∩ Υj
        ),
        name="sequence_consistency_lower",
    )

    # Constraint 11 & 12: Overtaking constraints
    # Loop over all aircraft pairs
    for i in P_list:
        for j in P_list:
            if i == j:
                continue

            for r_i in E[i]:
                edges_i = E[i][r_i]

                for r_j in E[j]:
                    edges_j = set(E[j][r_j])

                    for (u, v) in edges_i:
                        if (u, v) in edges_j:
                            model.addConstr(
                                Z[i, j, u] - Z[i, j, v] <= 2 -
                                (
                                    gp.quicksum(Gamma[i, r] for r in E[i] if (u, v) in E[i][r]) +
                                    gp.quicksum(Gamma[j, r] for r in E[j] if (u, v) in E[j][r])
                                ),
                                name=f"no_overtake_upper_{i}_{j}_{u}_{v}"
                            )

                            model.addConstr(
                                Z[i, j, u] - Z[i, j, v] >=
                                (
                                    gp.quicksum(Gamma[i, r] for r in E[i] if (u, v) in E[i][r]) +
                                    gp.quicksum(Gamma[j, r] for r in E[j] if (u, v) in E[j][r])
                                ) - 2,
                                name=f"no_overtake_lower_{i}_{j}_{u}_{v}"
                            )

    # Constraint 13 & 14: Head-on constraints (upper and lower)
    for i in P_list:
        for j in P_list:
            if i == j:
                continue
            for r_i in E[i]:
                edges_i = E[i][r_i]

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

    # Constraint 15: not scheduled before estimated touchdown time
    for j in A: 
        model.addConstr(
            
                gp.quicksum( t[j, first_node(route)] * Gamma[j, route]
                            for route in P.loc[P["id"] == j, "routes"].iloc[0]) 
                            >= P.loc[P["id"] == j, "ETD"].iloc[0],
            name="arrival_time_window_{j}"
        )

    # Constraint 16: Not scheduled before planned push-back time
    for i in D: 
        model.addConstr(
            
                gp.quicksum( t[i, first_node(route)] * Gamma[i, route]
                            for route in P.loc[P["id"] == i, "routes"].iloc[0]) 
                            >= P.loc[P["id"] == i, "PBT"].iloc[0],
            name="departure_time_window_{i}"
        )

    # Constraint 17,18 are not linearized: constraint 19,20 are linearized version
    # Constraint 19: max taxi speed
    for i in P_list:
        for r_i in E[i]:              # aircraft i's possible routes
            for (u, v) in E[i][r_i]:  # edges in route r_i
                model.addConstr(
                    t[i, v] - t[i, u] <= (length_edge((u, v)) / Suv_max) *
                    (M -M *(gp.quicksum(Gamma[i, r] 
                                        for r in routes_with_edge((u, v)))) +
                    gp.quicksum(Gamma[i, r] 
                                for r in routes_with_edge((u, v)))) ,
                    name=f"max_taxi_speed_{i}_{u}_{v}"
                )

    # Constraint 20: constrained by min taxi speed to enforce positive non-zero speed
        for i in P_list:
            for r_i in E[i]:              # aircraft i's possible routes
                for (u, v) in E[i][r_i]:  # edges in route r_i
                    model.addConstr(
                        t[i, v] - t[i, u] >= (length_edge((u, v)) / Suv_min) *
                        (M *(gp.quicksum(Gamma[i, r] 
                                            for r in routes_with_edge((u, v)))) - M +
                        gp.quicksum(Gamma[i, r] 
                                    for r in routes_with_edge((u, v)))) ,
                        name=f"min_taxi_speed_{i}_{u}_{v}"
                    )

    # Constraint 21,22 are non-linear

    # Constraint 23  
    for i in P_list:
        for j in P_list:
            if i == j:
                continue
            for r_i in E[i]:              # aircraft i's possible routes
                for (u, v) in E[i][r_i]:  # edges in route r_i
                    if u in U[((i, j))]:
                        model.addConstr(
                            t[j,u] - t[i,u] - (t[i,v] - t[i,u]) * (find_separation(i,j, Sep_scaled) / length_edge((u,v)))
                            >= - (3 - ((Z[i,j,u]) + gp.quicksum(Gamma[i, r] for r in routes_with_edge((u,v)) if r in P_routes[i])
                                        + gp.quicksum(Gamma[j, r] for r in P_routes[j] if u in route_nodes[r])
                                        )) * M,
                                            name="sep_situation1_{i}_{j}_{u}_{v}")
                                                                
            for r_j in E[j]:    
                for (w,v) in E[j][r_j]: # All possible edges in routes of j
                    if v in U[((i,j))]: 
                        model.addConstr(
                            t[i,v] - t[j,v] - (t[j,v] - t[j, w]) * find_separation(i,j, Sep_scaled) / length_edge((w, v)) >=
                                    - (3 - ((Z[j,i,v]) + 
                                            gp.quicksum(Gamma[j, r] for r in routes_with_edge((w,v)) if r in P_routes[j])
                                        + gp.quicksum(Gamma[i, r] for r in P_routes[i] if v in route_nodes[r]))) * M
                                            ,name="sep_situation2_{i}_{j}_{w}_{v}")

    # Constraint 28: Adds constraints for all routes in E
    for i in D:           
        for j in D:
            if i == j:
                continue

            # --- Add the constraint if both are departures ---
            model.addConstr(
                t[j,d] - t[i, d] - find_vortex_separation(i, j, V_scaled) >=
                - (1 - rho[i, j]) * M,
                name=f"vortex_sep_departure_{i}_{j}"
        )

    # Constraint 31
    for i in D:
        for j in A:
            # only consider nodes actually in j's route(s)
            for b_k in b:
                if b_k not in P.loc[P["id"] == j, "Upsilon"].iloc[0]:
                    continue  # skip nodes not in j's route

                model.addConstr(
                    t[j, b_k] - t[i, '17ra'] - Tidep >= - M * (1 - rho[i,j]),
                    name=f'runway_cross_arrival_{i}_{j}_{b_k}'
                )

    # Constraint 32
    for i in D:
        for j in A:
            # only consider nodes actually in j's route(s)
            for a_k in a:
                if a_k not in P.loc[P["id"] == j, "Upsilon"].iloc[0]:
                    continue  # skip nodes not in j's route

                model.addConstr(
                    t[i, '17ra'] - t[j, a_k] >= - M * (1 - rho[j,i]),
                    name=f'runway_cross_departure_{i}_{j}_{a_k}'
                )

    # Constraint 33
    for l in L:
        for i in A:
            for j in A:
                if i == j:
                    continue
                    
                if P.loc[P['id'] == i, "ETD"].iloc[0] <= P.loc[P['id'] == j, "ETD"].iloc[0]:
                    # i uses edge l ?
                    i_uses_l = any(l == e for route in E[i].values() for e in route)
                    # j uses edge l ?
                    j_uses_l = any(l == e for route in E[j].values() for e in route)

                    if i_uses_l and j_uses_l:
                        ETD_j = P.loc[P["id"] == j, "ETD"].iloc[0]

                        model.addConstr(
                            t[i, l[0]] <= ETD_j,
                            name=f"exit_capacity_{i}_{j}_{l}"
                        )

    model.optimize()
    print("Status:", model.status)
    print("Status name:", {2:"OPTIMAL", 3:"INFEASIBLE", 4:"INF_OR_UNBD", 5:"UNBOUNDED", 9:"TIME_LIMIT"}.get(model.status, "OTHER"))
    print("SolCount:", model.SolCount)

    model.write("model.lp")


    
    handles = {
    "Gamma": Gamma,
    "t": t,
    # "Z": {key : var.X for key, var in Z.items()},
    "R": R
    }
    return model, handles, P, P_list


params = {
    "M": 10000,
    "taxi_speed_multiplier": 1,
    "Tidep": 50,
    "sep_multiplier": 1.0,
    "vortex_multiplier": 1.0}

model, handles, P, P_list = build_model(params)

print(handles)



# For sensitivity analysis
__all__ = [
    "build_model",
    "P_list",
    "P",
    "get_first_node"
]


