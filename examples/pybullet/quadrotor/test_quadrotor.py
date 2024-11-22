#!/usr/bin/env python

from __future__ import print_function
import pybullet as p
import time
from examples.pybullet.utils.pybullet_tools.utils import add_data_path, connect, disconnect, wait_for_user, \
    draw_pose, Pose, Point, multiply, interpolate_poses, add_line, point_from_pose, remove_handles, BLUE

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

#####################################

def main():
    connect(use_gui=True)
    add_data_path()
    draw_pose(Pose(), length=1.)

    # Load the plane and the quadrotor model
    QUAD_PATH = "examples/pybullet/utils/models/quadrotor/quadrotor.urdf"
    quadrotor = p.loadURDF(QUAD_PATH, [0, 0, 0.5], useFixedBase=False)  # Adjust path as necessary
    
    # Check the model
    print("Loaded quadrotor:")
    print(f"Base position: {p.getBasePositionAndOrientation(quadrotor)}")

    # Start and end poses
    start_pose = ([0, 0, 0.5], [0, 0, 0, 1])  # Initial position and orientation (quaternion)
    end_pose = multiply(start_pose, Pose(Point(z=1.0)))  # Move 1 meter upwards
    
    print("Testing trajectory...")
    test_trajectory(quadrotor, start_pose, end_pose)

    # Disconnect
    wait_for_user("Press Enter to disconnect...")
    disconnect()

if __name__ == '__main__':
    main()
