import multiprocessing as mp

import matplotlib.pyplot as plt

import examples.ex_default_controller_model
import examples.ex_different_controllers

def _map(f):
    return f()

if __name__ == "__main__":

    funcs = [
        examples.ex_default_controller_model.ex_slow_leader,
        examples.ex_default_controller_model.ex_avg_speed_leader,
        examples.ex_default_controller_model.ex_fast_leader,
        examples.ex_default_controller_model.ex_def_controller_immobile_leader,
        examples.ex_default_controller_model.ex_def_controller_immobile_leader_transient,
        examples.ex_different_controllers.ex_avg_leader_half_acceleration_prediction_usual_distance_keeping,
        examples.ex_different_controllers.ex_avg_leader_half_acceleration_prediction_constant_distance_keeping,
        examples.ex_different_controllers.ex_slow_leader_constant_vel_to_next_car,
        examples.ex_different_controllers.ex_avg_speed_leader_constant_vel_to_next_car,
        examples.ex_different_controllers.ex_fast_leader_constant_vel_to_next_car,
        examples.ex_different_controllers.ex_avg_leader_zero_acceleration_prediction_usual_distance_keeping,
        examples.ex_different_controllers.ex_avg_leader_zero_acceleration_prediction_constant_distance_keeping,
        examples.ex_different_controllers.ex_avg_leader_half_acceleration_prediction_usual_distance_keeping,
        examples.ex_different_controllers.ex_avg_leader_half_acceleration_prediction_constant_distance_keeping,
        examples.ex_different_controllers.ex_avg_leader_mpc_horizon_1_sec_5_N,
        examples.ex_different_controllers.ex_avg_leader_mpc_horizon_10_sec_5_N,
    ]

    processes = [
        mp.Process(target=_map, args=(f,)) 
        for f in funcs
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()


    plt.show(block=True)
