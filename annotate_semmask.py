# Third Party
import numpy as np
import cv2

# In House


# Constants
IMG_SIZE = (320, 320, 3)
# Variables
drawing = False
mask = np.zeros(IMG_SIZE)
color = np.array([0,0,0])

def create_fixture_semantic_map():
    pass

# Mouse callback function
def draw_mask(event, x, y, flags, param):
    global mask, color

    if event == cv2.EVENT_LBUTTONDOWN:
        color = np.array([128,128,128])
        mask[y-10:y+10, x-10:x+10] = color

    elif event == cv2.EVENT_MBUTTONDOWN:
        mask[y-10:y+10,x-10:x+10] = np.array([255,255,255])

if __name__ == "__main__":
    # Read images
    # color_img = cv2.imread("/media/cklammer/KlammerData1/dev/f1tenth-project/output/20231210_183138/color_img/290.png")
    # color_img = cv2.resize(color_img, (IMG_SIZE[0], IMG_SIZE[1]))
    # depth_img = cv2.imread("/media/cklammer/KlammerData1/dev/f1tenth-project/output/20231210_183138/depth_img/290.png")
    # depth_img = cv2.resize(depth_img, (IMG_SIZE[0], IMG_SIZE[1]))

    # # Create a window and set the mouse callback
    # cv2.namedWindow("Image")
    # cv2.setMouseCallback("Image", draw_mask)

    # while True:
    #     cv2.imshow("Image", color_img.astype("uint8"))
    #     cv2.imshow("Interactive Mask", mask.astype("uint8"))

    #     key = cv2.waitKey(1) & 0xFF
    #     if key == 27:  # Press 'Esc' to exit
    #         break
    # cv2.imwrite("segmask.png", mask)

    # Visualization
    # norm_depth = cv2.normalize(depth_img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    # cv2.imshow("Depth", cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET))
    # cv2.waitKey()

    seg_color_img = cv2.imread("segmask.png")
    out_img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.int8)
    out_img[seg_color_img == 128] = 1
    out_img[seg_color_img == 255] = 2
    cv2.imwrite("seg_int.png", out_img)