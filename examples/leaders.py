import numpy as np

def velocity_relu_leader(t):
    return 

def fast_leader(t):
    v_points = ((0.0, 0.0), (0.5, 0.0), (2.5, 40.0), (4.0, 40.0), (7.0, 0.0))
    return _leader_state_piecewise_velocity_definition(t, v_points)

def avg_speed_leader(t):
    v_points = ((0.0, 0.0), (0.5, 0.0), (2.5, 20.0), (4.0, 20.0), (7.0, 0.0))
    return _leader_state_piecewise_velocity_definition(t, v_points)

def slow_leader(t):
    v_points = ((0.0, 0.0), (0.5, 0.0), (2.5, 5.0), (4.0, 5.0), (7.0, 0.0))
    return _leader_state_piecewise_velocity_definition(t, v_points)

def _leader_state_piecewise_velocity_definition(t: float, v_points: tuple[tuple[float, float]]):
    # leader velocity is a linear interpolation of the following points
    # a point is (t, v)
    pos_initial = 0

    pos = pos_initial
    # Before the specified points, velocity is constant
    first_t, first_v = v_points[0]
    if t < first_t:
        accel=0
        vel=first_v
        pos += (t-first_t)*vel
        return np.array[(pos, vel, accel)]
    
    prev_time = first_t
    prev_vel = first_v
    for t_p, v_p in v_points[1:]:
        accel = (v_p - prev_vel)/(t_p - prev_time)
        if t <= t_p:
            delta_t = t - prev_time
            vel = prev_vel + delta_t*accel
            pos += delta_t*prev_vel + delta_t**2 * accel / 2
            return np.array([pos, vel, accel])
        else:
            pos += (t_p - prev_time)*(prev_vel + v_p)/2
        prev_time = t_p
        prev_vel = v_p
    # if code reaches here, then time is after all points
    final_t, final_v = v_points[-1]
    accel=0
    vel=final_v
    pos += (t-final_t)*vel
    return np.array([pos, vel, accel])
