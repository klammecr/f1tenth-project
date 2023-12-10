# Third Party
import argparse
import cv2
from copy import deepcopy
import numpy as np
import scipy
from matplotlib import pyplot as plt
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog= "Waypoint Annotater",
                        description='This program takes in the map created by the SLAM toolbox and creates waypoints for pure pursuit')

    # Add arguments
    parser.add_argument("-m", "--map_file", default="waypoint_creation/aims7.pgm")
    parser.add_argument("-a", "--annotations", default=None)
    parser.add_argument("-o", "--out_path", default = "waypoint_creation/output")
    args = parser.parse_args()

    # Read the map
    map = cv2.imread(args.map_file)
    if args.annotations is None:
        
        # CREDIT: https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
        # ATTEMPT: MANUAL ANNOTATIONS
        annotations = []
        def draw_circle(event,x,y,flags,param):
            global mouseX,mouseY
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(map,(x,y),3,(0,0,255),-1)
                annotations.append((x,y))

        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle)

        while(1):
            cv2.imshow('image',map)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break

        # Save the raw annotations
        annotations = np.array(annotations)
        np.savetxt(f"{args.out_path}/annot_raw.txt", annotations)

        # Process the annotations with a spline then visualize
        coeff, param = scipy.interpolate.splprep([annotations[:,0], annotations[:, 1]], k = 2, s = 4)
        param_new = np.linspace(0, 1, 100)  # New parameter values
        x_ev, y_ev = scipy.interpolate.splev(param_new, coeff) # Use coefficients to generate a curve

        pts = np.stack((x_ev, y_ev)).astype("int")
        for x,y in pts.T.astype("int"):
            cv2.circle(map, (x,y), radius=3, color=(255,0,0))

        np.savetxt(f"{args.out_path}/annot_interp.txt", pts)
        cv2.imshow("Interpolated Spline Waypoints", map)
        cv2.waitKey()

        filt_annot = np.array([a for a in pts.T if np.all(map[a[1], a[0]] > 205)])

        dist_last_waypt = 0
        thresh = 25
        annot = filt_annot
        print(annot.shape)
        waypts = [annot[0]]
        for i in range(1, annot.shape[0], 1):
            dist_last_waypt += np.linalg.norm(annot[i] - annot[i-1])
            if dist_last_waypt >= thresh:
                waypts.append(annot[i])
                dist_last_waypt = 0

        print(waypts)
        waypts = np.array(waypts)
        for x,y in waypts:
            cv2.circle(map, (x,y), radius=3, color=(255,0,0))
        cv2.imshow("Filtered Waypoints", map)
        cv2.waitKey()
        waypts = np.hstack((waypts, np.zeros((waypts.shape[0], 1))))
        np.savetxt(f"{args.out_path}/final_waypoints.txt", waypts)

    # Post process

    # ATTEMPT: HARRIS CORNERS
    # Use image gradients in order to find corners
    # [IxIx  IxIy]
    # [IxIy  IyIy]
    # R = det(M) - k * tr(M)**2
    # Big values of R are corners
    # Negative values are edges
    # Small values are flat
    # map_gray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    # corners = cv2.cornerHarris(map_gray, 4, 3, k = 0.2)
    # # Curate the corners for display
    # corners = cv2.dilate(corners, None)
    # corner_vis[corners > 0.01*corners.max()] = [0, 0, 255]
    # cv2.imshow("Corners Visualization", corner_vis)
    # cv2.waitKey()