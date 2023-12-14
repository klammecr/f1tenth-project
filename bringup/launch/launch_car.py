"""Launch file for bringing up all nodes needed for operating the car."""

from launch import LaunchDescription
from launch_ros.actions import Node
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Initialize a new launch description.
    ld = LaunchDescription()
    
    # Get path to the params.yaml file.
    car_params = Path(get_package_share_directory("bringup"), "config", "car_params.yaml")

    # semantic_segmentation_node = Node(
    #     package="isaac_ros_image_segmentation",
    #     executable=
    # )

    # rrt_node = Node(
    #     package="lab6_pkg",
    #     executable="rrt_node.py",
    #     parameters=[bag_demo_params],
    #     remappings=[("costmap", "laser_local_costmap"),
    #                 ("pose", "ego_racecar/pose"),
    #                 ("path", "rrt_path")]
    # )
    # goal_publisher_node = Node(
    #     package="lab6_pkg",
    #     executable="goal_publisher_node.py",
    #     parameters=[bag_demo_params],
    #     remappings=[("pose", "ego_racecar/pose")]
    # )

    # ADD REALSENSE NODE.

    # Add the launch_ros "Node" actions we created.
    ld.add_action(rrt_node)
    ld.add_action(goal_publisher_node)

    # Return the newly created launch description.
    return ld