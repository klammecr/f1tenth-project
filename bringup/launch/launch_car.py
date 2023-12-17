"""Launch file for bringing up all nodes needed for operating the car."""

import os
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

    # ADD RRT NODE
    rrt_node = Node(
        package="project_rrt",
        executable="project_rrt",
        parameters=[os.path.join(
                get_package_share_directory('project_rrt'),
                'config', 'rrt_params.yaml')],
        name="project_rrt"
    )
    # ADD SEGBEV NODE
    seg_node = Node(
        package="segbev",
        executable="segbev_node",
        name="segbev"
    )

    # ADD REALSENSE NODE
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="realsense2"
    )

    # Add the launch_ros "Node" actions we created.
    ld.add_action(rrt_node)
    ld.add_action(realsense_node)
    ld.add_action(seg_node)

    # Return the newly created launch description.
    return ld