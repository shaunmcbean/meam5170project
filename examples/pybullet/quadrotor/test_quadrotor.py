#!/usr/bin/env python

from __future__ import print_function
import pybullet as p
import time
from math import *
import numpy as np
from examples.pybullet.utils.pybullet_tools.utils import add_data_path, connect, disconnect, wait_for_user, \
    draw_pose, Pose, Point, multiply, interpolate_poses, add_line, point_from_pose, remove_handles, BLUE

m = 0.5
g = 9.81
I = 0.004
a = 0.175

def test_trajectory(robot, start_pose, end_pose, step_size=0.01):
    """
    Simulates a trajectory for the quadrotor by interpolating between start and end poses.
    """
    handles = [add_line(point_from_pose(start_pose), point_from_pose(end_pose), color=BLUE)]
    pose_path = list(interpolate_poses(start_pose, end_pose, pos_step_size=step_size))
    
    for i, pose in enumerate(pose_path):
        print(f'Waypoint: {i + 1}/{len(pose_path)}')
        handles.extend(draw_pose(pose))
        p.resetBasePositionAndOrientation(robot, pose[0], pose[1])  # Update quadrotor pose
        time.sleep(0.1)  # Simulate real-time movement
    remove_handles(handles)

# def quadrotor_dynamics(x, u):
#     y, z, theta, y_dot, z_dot, theta_dot = x
#     u1, u2 = u

#     y_ddot = -np.sin(theta) / m * (u1 + u2)
#     z_ddot = -g + np.cos(theta) / m * (u1 + u2)
#     theta_ddot = a / I * (u1 - u2)

#     return np.array([y_dot, z_dot, theta_dot, y_ddot, z_ddot, theta_ddot])

def quadrotor_dynamics(x, u):
    # State unpacking
    x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot = x
    u1, u2, u3, u4 = u  # Thrust inputs for each propeller

    # Forces and torques
    thrust = np.array([0, 0, sum(u)])  # Net thrust (upward)
    moments = np.array([
        a * (u2 - u4),  # Roll torque
        a * (u3 - u1),  # Pitch torque
        I * (u1 - u2 + u3 - u4)  # Yaw torque
    ])

    # Rotation matrix from body to world frame
    R = np.array([
        [np.cos(psi) * np.cos(theta), np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi), np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)],
        [np.sin(psi) * np.cos(theta), np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi), np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)],
        [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]
    ])

    # Accelerations
    accel = R @ thrust / m - np.array([0, 0, g])  # Net acceleration

    # Angular accelerations (simplified for this case)
    angular_accel = moments / I

    # Return the state derivatives
    return np.array([
        x_dot, y_dot, z_dot,  # Position derivatives
        phi_dot, theta_dot, psi_dot,  # Angular velocity derivatives
        accel[0], accel[1], accel[2],  # Linear accelerations
        angular_accel[0], angular_accel[1], angular_accel[2]  # Angular accelerations
    ])


def simulate_quadrotor(quadrotor, xd_func, ud_func, t_f, dt):
    # initial state
    # x = np.array([0, 0.5, 0, 0, 0, 0])
    x = np.zeros(12)
    t = 0

    ############################
    # LQR controller
    Q = np.eye(12)
    Q[0, 0] = 10
    Q[1, 1] = 10
    Q[2, 2] = 10
    R = np.eye(4)
    Qf = Q

    LQR_controller = Quadcopter_LQR(Q, R, Qf, tf=t_f, x_d=xd_func, u_d=ud_func)
    ############################

    while t < t_f:
        ############################
        # desired state and input
        # xd = xd_func(t)
        # print(xd)
        # ud = ud_func(t)

        # # control input (LQR)
        # xe = x - xd
        # K = np.array([[5, 0, 0],    # Gain for y_error, z_error, theta_error
        #             [0, 10, 0]])
        # # Feedback control for x, y, z and angular errors
        # K_p = np.diag([2, 2, 5])  # Position gains
        # K_o = np.diag([0.1, 0.1, 0.1])   # Orientation gains

        # position_error = xe[:3]    # x, y, z errors
        # orientation_error = xe[3:6]  # roll, pitch, yaw errors

        # thrust_correction = -K_p @ position_error
        # moment_correction = -K_o @ orientation_error

        # # u_feedback = -K @ xe[:3]
        # # Convert corrections to propeller inputs
        # u_feedback = np.array([
        #     thrust_correction[2] / 4 + moment_correction[0] / (2 * a) + moment_correction[2] / (4 * I),
        #     thrust_correction[2] / 4 - moment_correction[0] / (2 * a) - moment_correction[2] / (4 * I),
        #     thrust_correction[2] / 4 + moment_correction[1] / (2 * a) + moment_correction[2] / (4 * I),
        #     thrust_correction[2] / 4 - moment_correction[1] / (2 * a) - moment_correction[2] / (4 * I)
        # ])

        # u = ud + u_feedback

        ############################
        # LQR Controller
        u = LQR_controller.compute_feedback(t, x)
        ############################

        u = np.clip(u, 0, 10)  # Clip inputs to feasible range

        # step dynamics
        x_dot = quadrotor_dynamics(x, u)
        x += x_dot * dt
        # PyBullet simulation update
        # position = [0, x[0], x[1]] # fixed x-axis motion (2D quadrotor)
        # orientation = p.getQuaternionFromEuler([0, 0, x[2]])
        # p.resetBasePositionAndOrientation(quadrotor, position, orientation)
        position = [x[0], x[1], x[2]]
        print(position)
        orientation = p.getQuaternionFromEuler([x[3], x[4], x[5]])
        p.resetBasePositionAndOrientation(quadrotor, position, orientation)
        p.stepSimulation()
        

        time.sleep(dt)
        t += dt
#####################################
# LQR Controller adapted from MEAM 5170 HW2 to 3D
from numpy import matmul
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp

class Quadcopter_LQR(object):
    '''
    Constructor. Compute function S(t) using S(t) = L(t) L(t)^t, by integrating backwards
    from S(tf) = Qf. We will then use S(t) to compute the optimal controller efforts in 
    the compute_feedback() function
    '''
    def __init__(self, Q, R, Qf, tf, x_d, u_d):
        self.m = m
        self.a = a
        self.I = I
        self.Q = Q
        self.R = R
        self.x_d = x_d
        self.u_d = u_d

        ''' 
        We are integrating backwards from Qf
        '''

        # Get L(tf) L(tf).T = S(tf) by decomposing S(tf) using Cholesky decomposition
        L0 = cholesky(Qf).transpose()

        # We need to reshape L0 from a square matrix into a row vector to pass into solve_ivp()
        l0 = np.reshape(L0, (144))
        # L must be integrated backwards, so we integrate L(tf - t) from 0 to tf
        initial_condition = [0, tf]
        sol = solve_ivp(self.dldt_minus, [0, tf], l0, dense_output=True)
        t = sol.t
        l = sol.y

        # Reverse time to get L(t) back in forwards time
        t = tf - t
        t = np.flip(t)
        l = np.flip(l, axis=1) # flip in time
        self.l_spline = interp1d(t, l)

    def Ldot(self, t, L):

        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot = self.x_d(t)
        u = self.u_d(t)
        Q = self.Q
        R = self.R

        F = np.sum(u)/self.m
        rx = cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)
        ry = cos(phi)*sin(theta)*sin(psi) + sin(phi)*cos(psi)
        rz = cos(phi)*cos(theta)

        A = np.zeros((12,12))
        A[0,6] = 1
        A[1,7] = 1
        A[2,8] = 1
        A[3,9] = 1
        A[4,10] = 1
        A[5,11] = 1

        A[6,3] = F*(-sin(phi)*sin(theta)*cos(psi) + cos(phi)*sin(psi))
        A[6,4] = F*(cos(phi)*cos(theta)*cos(psi))
        A[6,5] = F*(cos(phi)*sin(theta)*sin(psi) + sin(phi)*cos(psi))

        A[7,3] = F*(-sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi))
        A[7,4] = F*(cos(phi)*cos(theta)*sin(psi))
        A[7,5] = F*(cos(phi)*sin(theta)*cos(psi) - sin(phi)*sin(psi))

        A[8,3] = F*(-sin(phi)*cos(theta))
        A[8,4] = F*(-cos(phi)*sin(theta))

        B = np.zeros((12,4))
        B[6,:] = rx/self.m
        B[7,:] = ry/self.m
        B[8,:] = rz/self.m
        B[9,1] = 1/self.I
        B[9,3] = -1/self.I
        B[10,0] = -1/self.I
        B[10,2] = 1/self.I
        B[11,0] = self.a/self.I
        B[11,1] = -self.a/self.I
        B[11,2] = self.a/self.I
        B[11,3] = -self.a/self.I

        dLdt = np.zeros((12,12))
        # STUDENT CODE: compute d/dt L(t)
        dLdt = -0.5 * Q@inv(L).T-A.T@L + 0.5*L@L.T@B@inv(R)@ B.T@L

        return dLdt
    
    def dldt_minus(self, t, l):
        # reshape l to a square matrix
        L = np.reshape(l, (12, 12))

        # compute Ldot
        dLdt_minus = -self.Ldot(t, L)

        # reshape back into a vector
        dldt_minus = np.reshape(dLdt_minus, (144))
        return dldt_minus
    
    def compute_feedback(self, t, x):

        # Retrieve L(t)
        L = np.reshape(self.l_spline(t), (12, 12))
        
        x_e = x - self.x_d(t)
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot  = self.x_d(t)
        rx = cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)
        ry = cos(phi)*sin(theta)*sin(psi) + sin(phi)*cos(psi)
        rz = cos(phi)*cos(theta)

        B = np.zeros((12,4))
        B[6,:] = rx/self.m
        B[7,:] = ry/self.m
        B[8,:] = rz/self.m
        B[9,1] = 1/self.I
        B[9,3] = -1/self.I
        B[10,0] = -1/self.I
        B[10,2] = 1/self.I
        B[11,0] = self.a/self.I
        B[11,1] = -self.a/self.I
        B[11,2] = self.a/self.I
        B[11,3] = -self.a/self.I

        u_fb = -inv(self.R)@B.T@L@L.T@x_e
        # STUDENT CODE: Compute optimal feedback inputs u_fb using LQR
        # Add u_fb to u_d(t), the feedforward term. 
        # u = u_fb + u_d
        u = self.u_d(t) + u_fb
        return u
    
#####################################

def main():
    connect(use_gui=True)
    add_data_path()
    # draw_pose(Pose(), length=1.)

    # Load the plane and the quadrotor model
    QUAD_PATH = "examples/pybullet/utils/models/quadrotor/quadrotor.urdf"
    quadrotor = p.loadURDF(QUAD_PATH, [0, 0, 0.5], useFixedBase=False)  # Adjust path as necessary
    
    # Desired trajectory
    # def xd_func(t):
    #     z_d = 0.5 + 0.1 * t # Linear ascent
    #     return np.array([0, z_d, 0, 0, 0, 0])
    
    def xd_func(t):
        # x_d = 0.1
        # y_d = 0.1
        # z_d = 0.5 * t
        # return np.array([x_d, y_d, z_d, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        x_d = 0.1
        y_d = 0.1
        z_d = 0.5 * t
        z_dot = 0.1 * np.sin(2 * np.pi * t)
        return np.array([x_d, y_d, z_d, 0, 0, 0, 0, 0, z_dot, 0, 0, 0])

    
    # def ud_func(t):
    #     return np.array([m * g / 2, m * g / 2]) # Hover thrust inputs
    
    def ud_func(t):
        # hover_thrust = m * g / 4 + 0.1
        # return np.array([hover_thrust, hover_thrust, hover_thrust, hover_thrust])
        hover_thrust = m * g / 4
        additional_thrust = 0.1 * np.sin(2 * np.pi * t)  # Oscillating thrust
        return np.array([hover_thrust + additional_thrust] * 4)

    
    # Check the model
    print("Loaded quadrotor:")
    print(f"Base position: {p.getBasePositionAndOrientation(quadrotor)}")

    # Start and end poses
    # start_pose = ([0, 0, 0.5], [0, 0, 0, 1])  # Initial position and orientation (quaternion)
    # end_pose = multiply(start_pose, Pose(Point(z=1.0)))  # Move 1 meter upwards
    
    print("Testing trajectory...")
    # test_trajectory(quadrotor, start_pose, end_pose)

    simulate_quadrotor(quadrotor, xd_func, ud_func, t_f=5.0, dt=0.01)

    # Disconnect
    wait_for_user("Press Enter to disconnect...")
    disconnect()

if __name__ == '__main__':
    main()
