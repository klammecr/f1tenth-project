# Third Party
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import open3d as o3d

# In House
from roboteye.ground_robot import GroundRobot, Frames, COB

class SegBevNode(Node):
    def __init__(self):
        super().__init__("segbev_node")
        # TODO: Make into params
        self.INIT_RES_COL = (1920, 1080)
        self.INIT_RES_DEP = (640, 480)
        self.IMG_SIZE = (320, 320, 3)
        self.GRID_SIZE = (2, 3) # meters
        self.VOXEL_SIZE = (0.02, 0.02) # meters
        self.CAR_COLOR = (0,0,255)
        self.WALL_COLOR = (0,255,0)
        self.NEG_COLOR  = (128, 128, 128)
        self.NEG_IDX  = 0
        self.CAR_IDX  = 1
        self.WALL_IDX = 2

        # Latest image buffers
        self.color_img = None
        self.depth_img = None

        # TODO: Set up pubsub

    def bevify(self, voxel_grid):
        egocentric_grid = np.zeros((int(self.GRID_SIZE[0]/self.VOXEL_SIZE[0]), int(self.GRID_SIZE[1]/self.VOXEL_SIZE[1]), 3))
        origin_y = egocentric_grid.shape[0]-1
        origin_x = egocentric_grid.shape[1]//2
        voxels   = voxel_grid.get_voxels()
        for voxel in voxels:       
            # Find where the voxel is in the metric space
            metric_location = voxel_grid.origin+ voxel.grid_index*voxel_grid.voxel_size
            ego_loc = metric_location/voxel_grid.voxel_size
            lx, ly, lz = ego_loc

            if abs(ly) <= 10:

                # Convert to our ego-centric grid
                y_idx = origin_y-round(lz)
                x_idx = origin_x-round(lx)     

                egocentric_grid[y_idx, x_idx] += voxel.color/len(voxels)

    def bev_to_occ_grid(bev):
        occ_grid = (bev>0).astype("int")
        res = (cv2.morphologyEx(occ_grid.astype("uint8"), cv2.MORPH_OPEN, (3,3))).max(2)
        cv2.imshow("Occupancy Grid", res*255)
        cv2.waitKey()
        return res
    
    def draw_pts(self, pc, pts, color):
        # Create the pts
        pts_T = pts[:3].T
        pc_colors = np.stack([color for i in range(pts_T.shape[0])])

        # Merge
        comb_pts = np.vstack([pc.points, pts_T])
        comb_colors = np.vstack([pc.colors, pc_colors/255.])
        pc.points = o3d.utility.Vector3dVector(comb_pts)
        pc.colors = o3d.utility.Vector3dVector(comb_colors)

    def callback_image():
        pass


def main(args=None):
    rclpy.init(args=args)
    node = SegBevNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()