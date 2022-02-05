from __future__ import annotations

import matplotlib.pyplot as plt

from simulation.sim_dataflow import SimData


def plot_car_positions(data: SimData, include_leader=False) -> None:
    for i in range(data.n_cars):
        plt.plot(data.time, data.positions[i])
    if include_leader:
        plt.plot(data.time, [data.settings.leader_state(t)[0] for t in data.time], label="leader")
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