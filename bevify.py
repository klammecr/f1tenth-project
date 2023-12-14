# Third Party
import cv2
import numpy as np
from roboteye.geom import pi, inv_pi
from roboteye.ground_robot import GroundRobot, Frames, COB
import open3d as o3d

# In House

# CONSTANTS
INIT_RES_COL = (1920, 1080)
INIT_RES_DEP = (640, 480)
IMG_SIZE = (320, 320, 3)
GRID_SIZE = (2, 3) # meters
VOXEL_SIZE = (0.02, 0.02) # meters
colors_bgr = [(0,0,255), (0,255,0)]
colors_rgb = [(255,0,0), (0,255,0)]
CAR_IDX = 1
WALL_IDX = 2


def bevify(pcd, voxel_grid, K):
    egocentric_grid = np.zeros((int(GRID_SIZE[0]/VOXEL_SIZE[0]), int(GRID_SIZE[1]/VOXEL_SIZE[1]), 3))
    origin_y = egocentric_grid.shape[0]-1
    origin_x = egocentric_grid.shape[1]//2
    for voxel in voxel_grid.get_voxels():
        y_idx = origin_y-voxel.grid_index[2]
        x_idx = origin_x+voxel.grid_index[0]
        if np.sum(egocentric_grid[y_idx, x_idx]) == 0:
            egocentric_grid[y_idx, x_idx] = voxel.color
        else:
            if egocentric_grid[y_idx, x_idx, 0] > 0:
                egocentric_grid[y_idx, x_idx] = colors_bgr[CAR_IDX-1]
            else:
                egocentric_grid[y_idx, x_idx] = colors_bgr[WALL_IDX-1]
    cv2.imshow("BEV Semantic Grid", cv2.resize(egocentric_grid, (1024,1024)))
    cv2.waitKey()

def view_sem_img(sem_img):
    view_img = np.zeros_like(sem_img)
    wall_idxs = np.bitwise_and.reduce((seg_img_int == WALL_IDX), 2)
    car_idxs  = np.bitwise_and.reduce((seg_img_int == CAR_IDX), 2)
    view_img[wall_idxs] = colors_bgr[WALL_IDX-1]
    view_img[car_idxs]  = np.array(colors_bgr[CAR_IDX-1])
    cv2.imshow("Semantic Image", view_img.astype("uint8"))
    cv2.waitKey()

def draw_car_pts(pc, pts):
    pts_T = pts[:3].T
    pc.points = o3d.utility.Vector3dVector(pts_T)
    pc_colors = np.stack([colors_rgb[CAR_IDX-1] for i in range(pts_T.shape[0])])
    pc.colors = o3d.utility.Vector3dVector(pc_colors/255.)
    return pc

def draw_wall_pts(pc, pts):
    # Create the pts
    pts_T = pts[:3].T
    pc_colors = np.stack([colors_rgb[WALL_IDX-1] for i in range(pts_T.shape[0])])

    # Merge
    comb_pts = np.vstack([pc.points, pts_T])
    comb_colors = np.vstack([pc.colors, pc_colors/255.])
    pc.points = o3d.utility.Vector3dVector(comb_pts)
    pc.colors = o3d.utility.Vector3dVector(comb_colors)
    return pc

if __name__ == "__main__":
    # Read everything of interest
    depth_img = cv2.imread("/media/cklammer/KlammerData1/dev/f1tenth-project/output/20231210_183138/depth_img/296.png", -1)
    depth_img= depth_img/255
    cam_intr_dep = np.load("/media/cklammer/KlammerData1/dev/f1tenth-project/output/20231210_183138/depth_cam_info/depth_cam_info.npy")
    K = cam_intr_dep[0].reshape(3,3)
    depth_img = cv2.resize(depth_img, INIT_RES_DEP)
    seg_img_int = cv2.imread("seg_int.png")
    seg_img_int = cv2.resize(seg_img_int, INIT_RES_DEP)

    view_sem_img(seg_img_int)
    
    # Find where in the image the wall and car is
    locs_wall = np.where(seg_img_int == WALL_IDX)
    locs_car  = np.where(seg_img_int == CAR_IDX)
    pts_img_car = np.vstack((locs_car[1], locs_car[0]))
    pts_img_car_homog = np.vstack((pts_img_car, np.ones((1, pts_img_car.shape[1])))) * depth_img[locs_car[0], locs_car[1]]
    pts_img_wall = np.vstack((locs_wall[1], locs_wall[0]))
    pts_img_wall_homog = np.vstack((pts_img_wall, np.ones((1, pts_img_wall.shape[1])))) * depth_img[locs_wall[0], locs_wall[1]]

    # Selectively un-project each class into a 3D point cloud
    # gr = GroundRobot(cam_calib={"K":K}, cob=COB.NED_TO_CAM)
    # pts_car  = gr.transform_points(pts_img_car_homog.T, Frames.IMG_FRAME, Frames.BODY_FRAME_WORLD_ALIGNED)
    # pts_wall = gr.transform_points(pts_img_wall_homog.T, Frames.IMG_FRAME, Frames.BODY_FRAME_WORLD_ALIGNED)
    pts_car  = inv_pi(np.linalg.inv(K), np.eye(4), pts_img_car, depth_img[locs_car[0], locs_car[1]])
    pts_wall = inv_pi(np.linalg.inv(K), np.eye(4), pts_img_wall, depth_img[locs_wall[0], locs_wall[1]])

    # Create the point cloud to visualize    
    pc = o3d.geometry.PointCloud()
    pc = draw_car_pts(pc, pts_car)
    pc = draw_wall_pts(pc, pts_wall)
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, 0.02)
    #bevify(pc, voxel_grid,K)


    o3d.visualization.draw_geometries([voxel_grid], window_name="Combined Semantic PointCloud")