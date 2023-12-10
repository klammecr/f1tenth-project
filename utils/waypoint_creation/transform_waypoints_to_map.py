import argparse
import cv2
import numpy as np
import yaml

np.set_printoptions(suppress=True,precision=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog= "Waypoint Annotater",
                        description='This program takes in the map created by the SLAM toolbox and creates waypoints for pure pursuit')

    # Add arguments
    parser.add_argument("-m", "--map_file", default="waypoint_creation/aims7.pgm")
    parser.add_argument("-w", "--waypoints", default="waypoint_creation/output/final_waypoints.txt")
    parser.add_argument("-y", "--yaml", default="waypoint_creation/aims7.yaml")
    parser.add_argument("-v", "--velocity", default=3.0)
    args = parser.parse_args()

    # Read image
    map = cv2.imread(args.map_file)
    
    h, w, _ = map.shape
    waypts = np.loadtxt(args.waypoints)
    waypts[:, -1] = 1

    # Parse yaml file
    yaml_data = None
    with open(args.yaml, "r") as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    if yaml_data is not None:
        origin = np.array(yaml_data["origin"])
        resolution = yaml_data["resolution"]
    else:
        origin = np.array([0,0])
        resolution = 0.05

    # Distance in meters from the world center to the top left of the image
    T_map_origin = np.eye(3)
    T_map_origin[:2, -1] =  np.array([0, resolution*h]) + origin[:2]

    # Rotate to the map
    R_map = np.eye(3)
    R_map[0, :2] = np.array([1, 0])
    R_map[1, :2] = np.array([0, -1])

    waypts[:, :2] *= resolution
    out_pts = (T_map_origin @ R_map @ waypts.T).T
    
    # Set the desired yaw of each waypoint, just make it constant velocity for now
    yaws = np.zeros((out_pts.shape[0], 1))
    prev_yaw = 0
    velocity = np.zeros((out_pts.shape[0], 1))
    for i in range(0, out_pts.shape[0]-1):
        # Translation vector
        delta = out_pts[i+1] - out_pts[i]

        # Arctan of translation vector is theta
        yaws[i] = np.arctan2(delta[1], delta[0])

        velocity[i] = 2.5

        prev_yaw = yaws[i]

    # Finalize waypoints
    print(out_pts)
    final_waypts = np.hstack((out_pts[:, :2], yaws, velocity))

    map_name = args.map_file.split("/")[-1]
    np.savetxt(f"{map_name}_transformed.csv", final_waypts, delimiter=" ", fmt='%f')
    