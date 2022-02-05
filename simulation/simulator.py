from __future__ import annotations

import numpy as np

from control.controlled_car import ControlledCar
from control.distributed_controller import DistributedControlledSystem
from simulation.sim_dataflow import SimSettings, SimData

class Simulator:
    def __init__(self, settings: SimSettings) -> None:
        self.settings = settings

        self.system = DistributedControlledSystem(
            self.settings.distributed_controller_params,
            cars=tuple((
                ControlledCar(car_params, controller_params)
                for car_params, controller_params in self.settings.cars_params
            ))
        )
        self.simulated = False

    def step(self, ref: float, time_step: float) -> None:
        self.system.step(ref, time_step)

    def simulate(self) -> None:
        for t in np.arange(0, self.settings.time_sim, self.settings.step_sim):
            self.step(self.settings.leader_pos(t), self.settings.step_sim)

        self.simulated = True

    def get_sim_data(self) -> SimData:
        if not self.simulated:
            raise RuntimeError("Simulation not run yet")

        for car in self.system.cars:
            car.states = np.array(car.states)

        return SimData(
            settings=self.settings,
            n_cars=len(self.system.cars),
            time=np.arange(0, self.settings.time_sim+self.settings.step_sim, self.settings.step_sim),
            positions=np.array([car.states[:, 0] for car in self.system.cars]),
            velocities=np.array([car.states[:, 1] for car in self.system.cars]),
            accelerations=np.array([car.states[:, 2] for car in self.system.cars]),
            control_signals=np.array([car.control_signals for car in self.system.cars]),
            references=np.array(self.system.references)
        )

if __name__ == "__main__":
    from control.car_controller import ControllerParameters
    from control.distributed_controller import DistributedControllerParameters
    from dynamics.car import CarParameters

    # example simulation with two cars
    settings = SimSettings(
        step_sim=0.1,
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
        distributed_controller_params= DistributedControllerParameters(),
        leader_pos=lambda t: t
    )

    sim = Simulator(settings)
    sim.simulate()

    print(sim.get_sim_data())
