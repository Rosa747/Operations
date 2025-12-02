import pandas as pd
import matplotlib.pyplot as plt

from Operations_assignment import (
    build_model,
    get_first_node
)

def extract_solution(handles, P, P_list):
    """Extract aircraft route choices, times, delays, etc."""
    Gamma = handles["Gamma"]
    t     = handles["t"]
    Z     = handles["Z"]
    R     = handles["R"]

    results = []

    aircraft_list = sorted(set(i for (i, _) in t.keys()))

    for ac in aircraft_list:

        chosen_route = None
        for (i, r) in Gamma.keys():
            if i == ac and Gamma[i, r].X > 0.5:
                chosen_route = r

        node_times = {
            u: t[ac, u].X
            for (_, u) in t.keys()
            if _ == ac
        }

        if ac in P_list:
            ad_flag = P.loc[P["id"] == ac, "A/D"].iloc[0]

            if ad_flag == "A":  
                sched_time = P.loc[P["id"] == ac, "ETD"].iloc[0]
            else:               
                sched_time = P.loc[P["id"] == ac, "PBT"].iloc[0]

            start_node = get_first_node(ac, P, R, handles["Gamma"])

            delay = node_times[start_node] - sched_time
        else:
            delay = None

        results.append({
            "aircraft": ac,
            "chosen_route": chosen_route,
            "delay": delay,
            "node_times": node_times
        })

    return results

def run_sensitivity(param_name, values, base_params):
    master = []

    for val in values:
        print(f"\n=== Running scenario: {param_name} = {val} ===")

        params = base_params.copy()
        params[param_name] = val

        model, handles, P, P_list = build_model(params)
        model.optimize()

        entry = {
            "param": param_name,
            "value": val,
            "status": model.Status,
            "objective": model.ObjVal if model.Status == 2 else None
        }

        if model.Status == 2:   
            sol = extract_solution(handles, P, P_list)

            for ac_data in sol:
                row = entry.copy()
                row["aircraft"] = ac_data["aircraft"]
                row["chosen_route"] = ac_data["chosen_route"]
                row["delay"] = ac_data["delay"]
                master.append(row)

            pd.DataFrame(sol).to_csv(
                f"scenario_{param_name}_{val}.csv",
                index=False
            )

    df = pd.DataFrame(master)
    df.to_csv(f"sensitivity_{param_name}_full.csv", index=False)

    return df

# Plot
def plot_sensitivity(df, param_name):
    """Plot objective value vs parameter value."""
    pivot = df.groupby("value")["objective"].mean()

    plt.figure(figsize=(7, 4))
    plt.plot(pivot.index, pivot.values, marker="o")
    plt.xlabel(param_name)
    plt.ylabel("Objective Value")
    plt.grid(True)
    plt.title(f"Sensitivity of Objective to {param_name}")
    plt.tight_layout()
    plt.savefig(f"sensitivity_plot_{param_name}.png")
    plt.show()


# Run sensitivity
if __name__ == "__main__":

    base_params = {
        "separation_multiplier": 1.0,
        "vortex_multiplier": 1.0,
        "taxi_speed_multiplier": 1.0,
        "Tidep": 55,
        "M": 1e4
    }

    # 1. Sensitivity: Separation Minima Sep
    sep_values = [0.8, 1.0, 1.2, 1.4]
    df_sep = run_sensitivity("separation_multiplier", sep_values, base_params)
    plot_sensitivity(df_sep, "separation_multiplier")

    # 2. Sensitivity: Separation Minima V
    V_values = [0.8, 1.0, 1.2, 1.4]
    df_V = run_sensitivity("vortex_multiplier", V_values, base_params)
    plot_sensitivity(df_V, "vortex_multiplier")

    # 3. Sensitivity: Taxi Speed
    speed_values = [0.8, 1.0, 1.2]
    df_speed = run_sensitivity("taxi_speed_multiplier", speed_values, base_params)
    plot_sensitivity(df_speed, "taxi_speed_multiplier")

    # 4. Sensitivity: Departure Window (Tidep)
    tidep_values = [40, 55, 70]
    df_tidep = run_sensitivity("Tidep", tidep_values, base_params)
    plot_sensitivity(df_tidep, "Tidep")
