import numpy as np
from rot_utils import *
from gym.envs.mujoco import *
import numpy as np


def sim_cartpole(x0, u, dt):
    DT, t = 0.1, 0
    def dynamics(x, u):
        mc, mp, l, g, I = 10.0, 1.0, 0.5, 9.81, 0.25
        s, c = np.sin(x[1]), np.cos(x[1])
        xddot = (u + mp * s *(l * np.square(x[3]) + g * c)) / (mc + mp * np.square(s))
        tddot = (-u * c - mp * l * np.square(x[3]) * c * s 
                 -(mc + mp) * g * s) / (l * (mc + mp * np.square(s)))
        return np.concatenate([x[2:4], xddot, tddot])
        
    while t < dt:
        current_dt = min(DT, dt-t)
        x0 = x0 + current_dt * dynamics(x0, u)
        t += current_dt
    
    return x0


def sim_heli(x0, u0, total_dt):
    DT, t = 0.05, 0
    ####################################
    #set up idx, model params, features#
    ####################################
    idx, model = dict(), dict()
    k = 0
    keys = ["ned_dot", "ned", "pqr", "axis_angle"]
    for ky in range(len(keys)):
        idx[keys[ky]] = np.arange(k,k+3)
        k += 3
        
    keys = ["m", "Ixx", "Iyy", "Izz", "Tx", "Ty", "Tz", "Fx", "Fy", "Fz"]
    values = [5, .3, .3, .3, np.array([0, -1.0410, 3.9600]), 
             np.array([0, -0.9180, -2.7630]), np.array([0, -0.7740, 4.4520]),
             np.array([-0.2400]), np.array([-3.0, -0.6000]), np.array([0, -0.0025, -137.5000])]
    for ky in range(len(keys)):
        model[keys[ky]] = values[ky]
        
    def compute_forces_and_torques(x0, u0, model, idx):
        # compute helicopter velocity in its own frame 
        # (it experiences drag forces in its own frame)
        uvw = express_vector_in_quat_frame(x0[idx['ned_dot']], quaternion_from_axis_rotation(x0[idx['axis_angle']]))
        ## aerodynamic forces 

        # expressed in heli frame:
        Fxyz_minus_g = np.zeros(3)
        Fxyz_minus_g[0] = np.dot(model['Fx'], uvw[0])
        Fxyz_minus_g[1] = np.dot(model['Fy'], np.array([1, uvw[1]]))
        Fxyz_minus_g[2] = np.dot(model['Fz'], np.array([1, uvw[2], u0[3]]))

        # expressed in ned frame
        F_ned_minus_g = rotate_vector(Fxyz_minus_g, quaternion_from_axis_rotation(x0[idx['axis_angle']]))

        # add gravity to complete the forces:
        Fned = F_ned_minus_g + np.dot(model['m'], np.array([0, 0, 9.81]))

        ## torques
        Txyz = np.zeros(3)
        Txyz[0] = np.dot(model['Tx'], np.array([1, x0[idx['pqr'][0]], u0[0]]))
        Txyz[1] = np.dot(model['Ty'], np.array([1, x0[idx['pqr'][1]], u0[1]]))
        Txyz[2] = np.dot(model['Tz'], np.array([1, x0[idx['pqr'][2]], u0[2]]))
        return Fned, Txyz
        
    x1 = np.zeros_like(x0)
    while t < total_dt:
        dt = min(DT, total_dt-t)
        # compute forces and torques
        Fned, Txyz = compute_forces_and_torques(x0, u0, model, idx)
        
        # forward integrate state            
        # angular rate and velocity simulation:  [this ignores inertial coupling;
        # apparently works just fine on our helicopters]

        x1[idx['ned_dot']] = x0[idx['ned_dot']] + dt * Fned / model['m']
        x1[idx['pqr']] = x0[idx['pqr']] + dt * Txyz / np.array([model['Ixx'], model['Iyy'], model['Izz']])


        # position and orientation merely require integration (we use Euler integration):
        x1[idx['ned']] = x0[idx['ned']] + dt * x0[idx['ned_dot']]
        x1[idx['axis_angle']] = axis_angle_dynamics_update(x0[idx['axis_angle']], x0[idx['pqr']]*dt)

        x0 = x1
        t += dt
        
    return x0

