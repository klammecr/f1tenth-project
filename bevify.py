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
CAR_COLOR = (0,0,255)
WALL_COLOR = (0,255,0)
NEG_COLOR  = (128, 128, 128)
NEG_IDX  = 0
CAR_IDX  = 1
WALL_IDX = 2


def bevify(voxel_grid):
    egocentric_grid = np.zeros((int(GRID_SIZE[0]/VOXEL_SIZE[0]), int(GRID_SIZE[1]/VOXEL_SIZE[1]), 3))
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
    


    # See what the winning class is
    majority_class = egocentric_grid.argmax(axis=2)
    egocentric_grid[majority_class == 2]  = CAR_COLOR
    egocentric_grid[majority_class == 1] = WALL_COLOR
    # egocentric_grid[majority_class == 0] = NEG_COLOR
    # Vis
    cv2.imshow("BEV Semantic Grid", cv2.resize(egocentric_grid, (1024,1024)))
    cv2.waitKey()
    return egocentric_grid

def bev_to_occ_grid(bev):
    occ_grid = (bev>0).astype("int")
    res = (cv2.morphologyEx(occ_grid.astype("uint8"), cv2.MORPH_OPEN, (3,3))).max(2)
    cv2.imshow("Occupancy Grid", res*255)
    cv2.waitKey()
    return res

def view_sem_img(sem_img):
    view_img = np.zeros_like(sem_img)
    wall_idxs = np.bitwise_and.reduce((seg_img_int == WALL_IDX), 2)
    car_idxs  = np.bitwise_and.reduce((seg_img_int == CAR_IDX), 2)
    view_img[wall_idxs] = WALL_COLOR
    view_img[car_idxs]  = CAR_COLOR
    cv2.imshow("Semantic Image", view_img.astype("uint8"))
    cv2.waitKey()

def draw_car_pts(pc, pts):
    pts_T = pts[:3].T
    pc.points = o3d.utility.Vector3dVector(pts_T)
    pc_colors = np.stack([CAR_COLOR for i in range(pts_T.shape[0])])
    pc.colors = o3d.utility.Vector3dVector(pc_colors/255.)
    return pc

def draw_wall_pts(pc, pts):
    # Create the pts
    pts_T = pts[:3].T
    pc_colors = np.stack([WALL_COLOR for i in range(pts_T.shape[0])])

    # Merge
    comb_pts = np.vstack([pc.points, pts_T])
    comb_colors = np.vstack([pc.colors, pc_colors/255.])
    pc.points = o3d.utility.Vector3dVector(comb_pts)
    pc.colors = o3d.utility.Vector3dVector(comb_colors)
    return pc

def draw_neg_pts(pc, pts):
    # Create the pts
    pts_T = pts[:3].T
    pc_colors = np.stack([NEG_COLOR for i in range(pts_T.shape[0])])

    # Merge
    comb_pts = np.vstack([pc.points, pts_T])
    comb_colors = np.vstack([pc.colors, pc_colors/255.])
    pc.points = o3d.utility.Vector3dVector(comb_pts)
    pc.colors = o3d.utility.Vector3dVector(comb_colors)
    return pc

def process(seg_img):
    # Find where in the image the wall and car is
    locs_wall = np.where(seg_img == WALL_IDX)
    locs_car  = np.where(seg_img == CAR_IDX)
    locs_neg  = np.where(seg_img == 0)
    pts_img_car = np.vstack((locs_car[1], locs_car[0]))
    pts_img_car_homog = np.vstack((pts_img_car, np.ones((1, pts_img_car.shape[1])))) * depth_img[locs_car[0], locs_car[1]]
    pts_img_wall = np.vstack((locs_wall[1], locs_wall[0]))
    pts_img_wall_homog = np.vstack((pts_img_wall, np.ones((1, pts_img_wall.shape[1])))) * depth_img[locs_wall[0], locs_wall[1]]
    pts_img_neg = np.vstack((locs_neg[1], locs_neg[0]))
    pts_img_neg_homog = np.vstack((pts_img_neg, np.ones((1, pts_img_neg.shape[1])))) * depth_img[locs_neg[0], locs_neg[1]]

    # Selectively un-project each class into a 3D point cloud
    #gr = GroundRobot(cam_calib={"K":K}, cob=COB.NED_TO_CAM)
    gr = GroundRobot(cam_calib={"K":K})
    pts_car  = gr.transform_points(pts_img_car_homog.T, Frames.IMG_FRAME, Frames.BODY_FRAME)
    pts_wall = gr.transform_points(pts_img_wall_homog.T, Frames.IMG_FRAME, Frames.BODY_FRAME)
    pts_neg  = gr.transform_points(pts_img_neg_homog.T, Frames.IMG_FRAME, Frames.BODY_FRAME) 


    # Create the point cloud to visualize
    pc = o3d.geometry.PointCloud()
    pc = draw_car_pts(pc, pts_car)
    pc = draw_wall_pts(pc, pts_wall)
    #pc = draw_neg_pts(pc, pts_neg)
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, 0.02)
    o3d.visualization.draw_geometries([voxel_grid], window_name="Combined Semantic PointCloud")
    bev = bevify(voxel_grid)
    occ_grid = bev_to_occ_grid(bev)
    return occ_grid


if __name__ == "__main__":
    # Read everything of interest
    #depth_img = cv2.imread("/media/cklammer/KlammerData1/dev/f1tenth-project/output/20231210_183138/depth_img/136.png", -1).astype("float32")
    depth_img = cv2.imread("/media/cklammer/KlammerData1/dev/f1tenth-project/output/20231210_183138/depth_img/291.png", -1).astype("float32")
    cam_intr_dep = np.load("/media/cklammer/KlammerData1/dev/f1tenth-project/output/20231210_183138/depth_cam_info/depth_cam_info.npy")
    K = cam_intr_dep[0].reshape(3,3)
    depth_img = cv2.resize(depth_img, INIT_RES_DEP)
    seg_img_int = cv2.imread("seg_int.png")
    seg_img_int = cv2.resize(seg_img_int, INIT_RES_DEP)

    # view_sem_img(seg_img_int)
    
    process(seg_img_int)


    