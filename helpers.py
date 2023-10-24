import cv2
import matplotlib.pyplot as plt


def visualise(img_left, img_right, text=None, wait=1, backend="cv2"):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2

    img_left = cv2.putText(img_left, "img_left", (20, img_left.shape[0]-20), font, fontScale, color, thickness, cv2.LINE_AA)
    img_right = cv2.putText(img_right, "img_right", (20, img_right.shape[0]-20), font, fontScale, color, thickness, cv2.LINE_AA)
    
    img_vis = cv2.hconcat([img_left, img_right])
    if text is not None:
        img_vis = cv2.putText(img_vis, text, (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)

        

    if backend == "cv2":
        cv2.imshow('visualise', img_vis)

        if wait is not None:
            if cv2.waitKey(wait) == ord('q'):
                pass
    
    else:
        img_vis = img_vis[...,::-1]
        plt.figure(dpi=300)
        plt.axis('off')
        plt.imshow(img_vis)
