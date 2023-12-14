# Third Party
import numpy as np
import cv2
import open3d as o3d

# ROS Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid

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
        self.depth_img = None
        self.K = None

        # Set up pubsub
        self.sem_sub = self.create_subscription(
            Image,
            "/color/image_raw",
            self.seg_img_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/depth/image_rect_raw",
            self.depth_img_callback,
            10
        )
        self.depth_cam_info_sub = self.create_subscription(
            CameraInfo,
            "/depth/camera_info",
            self.depth_cam_info_callack,
            10
        )

        self.occ_grid_pub = self.create_publisher(
            OccupancyGrid,
            "/occ_grid",
            10
        )

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
        return egocentric_grid

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

    def callback_seg_img(self, msg : Image):
        # Extract out seg img
        seg_img = msg.data.reshape(msg.height, msg.width)

        # Find where in the image the wall and car is
        locs_wall = np.where(seg_img == self.WALL_IDX)
        locs_car  = np.where(seg_img == self.CAR_IDX)
        locs_neg  = np.where(seg_img == 0)
        pts_img_car = np.vstack((locs_car[1], locs_car[0]))
        pts_img_car_homog = np.vstack((pts_img_car, np.ones((1, pts_img_car.shape[1])))) * self.depth_img[locs_car[0], locs_car[1]]
        pts_img_wall = np.vstack((locs_wall[1], locs_wall[0]))
        pts_img_wall_homog = np.vstack((pts_img_wall, np.ones((1, pts_img_wall.shape[1])))) * self.depth_img[locs_wall[0], locs_wall[1]]
        pts_img_neg = np.vstack((locs_neg[1], locs_neg[0]))
        pts_img_neg_homog = np.vstack((pts_img_neg, np.ones((1, pts_img_neg.shape[1])))) * self.depth_img[locs_neg[0], locs_neg[1]]

        # Selectively un-project each class into a 3D point cloud
        gr = GroundRobot(cam_calib={"K":self.K})
        pts_car  = gr.transform_points(pts_img_car_homog.T, Frames.IMG_FRAME, Frames.BODY_FRAME)
        pts_wall = gr.transform_points(pts_img_wall_homog.T, Frames.IMG_FRAME, Frames.BODY_FRAME)
        pts_neg  = gr.transform_points(pts_img_neg_homog.T, Frames.IMG_FRAME, Frames.BODY_FRAME) 


        # Create the point cloud to visualize
        pc = o3d.geometry.PointCloud()
        pc = self.draw_car_pts(pc, pts_car)
        pc = self.draw_wall_pts(pc, pts_wall)
        #pc = draw_neg_pts(pc, pts_neg)
        
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, 0.02)
        # o3d.visualization.draw_geometries([voxel_grid], window_name="Combined Semantic PointCloud")
        bev = self.bevify(voxel_grid)
        occ_grid = self.bev_to_occ_grid(bev)

        # Create an occupancy grid message
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.frame_id = 'map'
        occupancy_grid.info.width = occ_grid.shape[1]  # Set the width of the grid
        occupancy_grid.info.height = occ_grid.shape[0]  # Set the height of the grid
        occupancy_grid.info.resolution = self.VOXEL_SIZE[0]  # Set the resolution of each grid cell
        self.occ_grid_pub.publish(occupancy_grid)

    def depth_cam_info_callack(self, msg: CameraInfo):
        self.K = msg.k.reshape(3,3)

    def callback_depth_img(self, msg : Image):
        self.depth_img = msg.data.reshape(msg.height, msg.width)
        self.depth_img *= 1e-3

def main(args=None):
    rclpy.init(args=args)
    node = SegBevNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()