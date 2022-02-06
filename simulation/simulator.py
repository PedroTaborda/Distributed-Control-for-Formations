from __future__ import annotations

import time

import numpy as np

from control.controlled_car import ControlledCar
from control.distributed_system import DistributedControlledSystem
from simulation.sim_dataflow import SimSettings, SimData

class Simulator:
    def __init__(self, settings: SimSettings) -> None:
        self.settings = settings

        self.system = DistributedControlledSystem(
            cars=tuple((
                ControlledCar(car_params, controller_params)
                for car_params, controller_params in self.settings.cars_params
            ))
        )

        self.sim_t = 0.0

    def step(self, ref: float, time_step: float) -> None:
        self.system.step(ref, time_step)

    def simulate(self) -> None:
        for t in np.arange(0, self.settings.time_sim, self.settings.controller_sample_time):
            t0 = time.time()
            self.step(self.settings.leader_state(t), self.settings.controller_sample_time)
            print(f"======= Time step: {time.time() - t0:.2f} ({t:.2f}/{self.settings.time_sim:.2f})")
            self.sim_t = t 


    def get_sim_data(self) -> SimData:
        for car in self.system.cars:
            car.states = np.array(car.states)
        
        time = np.linspace(0, self.sim_t, len(self.system.cars[0].states), endpoint=False)
        return SimData(
            settings=self.settings,
            n_cars=len(self.system.cars),
            time=time,
            positions=np.array([car.states[:, 0] for car in self.system.cars]),
            velocities=np.array([car.states[:, 1] for car in self.system.cars]),
            accelerations=np.array([car.states[:, 2] for car in self.system.cars]),
            control_signals=np.array([car.control_signals for car in self.system.cars]),
            references=np.array(self.system.references)
        )

if __name__ == "__main__":
    from control.car_controller import ControllerParameters
    from dynamics.car import CarParameters

    # example simulation with two cars
    settings = SimSettings(
        controller_sample_time=0.1,
        time_sim=100.0,
        cars_params=(
            (
                CarParameters(),
                ControllerParameters()
            ),
            (
                CarParameters(),
                ControllerParameters()
            )
        ),
        leader_state=lambda t: t
    )

    sim = Simulator(settings)
    sim.simulate()

    print(sim.get_sim_data())
