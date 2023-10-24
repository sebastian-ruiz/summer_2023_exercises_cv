import os
import sys
import cv2
from timeit import default_timer as timer
from img_loader import ImgLoader
from helpers import visualise
from rich import print

    
if __name__ == '__main__':

    img_loader = ImgLoader(use_ros=True)

    is_record = True
    is_visualise = True
    record_all_frames = False

    if is_record:
        recording_path = "recording"
        path_exists = True

        counter = 1
        while path_exists:
            if os.path.exists(f"{recording_path}{counter}"):
                path_exists = True
            else:
                path_exists = False
                recording_path = f"{recording_path}{counter}"

            counter += 1

        if not os.path.exists(recording_path):
            print(f"[green]creating recording path: {recording_path}")
            os.makedirs(os.path.join(recording_path, "LEFT"))
            os.makedirs(os.path.join(recording_path, "RIGHT"))

    counter = 0
    start = timer()

    for img_left, img_right, gt in img_loader.next():
        end = timer()
        print(f"fps: {1/(end - start)}")
        start = end
        
        if img_left is not None and img_right is not None:
            if is_record:
                if record_all_frames or counter % 10 == 0:
                    cv2.imwrite(f"{recording_path}/LEFT/frame-{counter}.jpg", img_left)
                    cv2.imwrite(f"{recording_path}/RIGHT/frame-{counter}.jpg", img_right)

            if is_visualise:
                visualise(img_left, img_right, "", wait=1)

            counter += 1

    if is_visualise:
        cv2.destroyAllWindows()
