from __future__ import annotations

import matplotlib.pyplot as plt

from simulation.sim_dataflow import SimData


def plot_car_positions(data: SimData, include_leader: bool = False, plot_references: bool = True) -> None:
    # fill between car back and front
    for i in range(data.n_cars):
        color = "C" + str(i)
        plt.fill_between(
            data.time, 
            data.positions[:, i],
            data.positions[:, i] + data.settings.cars_params[i][0].length, 
            alpha=0.8,
            color=color
        )
        if plot_references:
            plt.plot(
                data.time,
                data.references[:, i],
                color=color,
                linestyle="--",
            )
    

    if include_leader:
        plt.fill_between(
            data.time, 
            [data.settings.leader_state(t)[0] for t in data.time],
            [data.settings.leader_state(t)[0] + data.settings.cars_params[0][0].length for t in data.time], 
            alpha=0.8, 
            label="leader", 
            color="black"
        )
    
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.legend()

    
def plot_car_reference_error(data: SimData) -> None:
    for i in range(data.n_cars):
        plt.plot(data.time, data.references[i] - data.positions[i])
    plt.xlabel("time (s)")
    plt.ylabel("reference error (m)")

def plot_intercar_distances(data: SimData) -> None:
    for i in range(data.n_cars):
        for j in range(i + 1, data.n_cars):
            plt.plot(data.time, data.positions[i] - data.positions[j], label=f"{i}-{j}")