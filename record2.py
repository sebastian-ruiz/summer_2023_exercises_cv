import os
import rospy

import std_msgs.msg
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from timeit import default_timer as timer
import numpy as np
import cv2
import message_filters
import queue as Queue
from rich import print
from helpers import visualise

class Recorder():
    def __init__(self) -> None:
        rospy.init_node("client")
        rospy.loginfo('start')

        self.time1 = timer()

        self.is_record = True
        self.is_visualise = True
        self.record_all_frames = False

        if self.is_record:
            self.recording_path = "recording"
            path_exists = True

            counter = 1
            while path_exists:
                if os.path.exists(f"{self.recording_path}{counter}"):
                    path_exists = True
                else:
                    path_exists = False
                    self.recording_path = f"{self.recording_path}{counter}"

                counter += 1

            if not os.path.exists(self.recording_path):
                print(f"[green]creating recording path: {self.recording_path}")
                os.makedirs(os.path.join(self.recording_path, "LEFT"))
                os.makedirs(os.path.join(self.recording_path, "RIGHT"))

        self.counter = 0

        image0 = message_filters.Subscriber("/video_stream/img_cam0", CompressedImage)
        image1 = message_filters.Subscriber("/video_stream/img_cam1", CompressedImage)
        ts = message_filters.TimeSynchronizer([image0, image1], queue_size=5)
        ts.registerCallback(self.img_callback)
    
    def img_callback(self, msg1, msg2):
        # print("callback")

        np_arr1 = np.fromstring(msg1.data, np.uint8)
        np_arr2 = np.fromstring(msg2.data, np.uint8)
        image_np1 = cv2.imdecode(np_arr1, cv2.IMREAD_COLOR)
        image_np2 = cv2.imdecode(np_arr2, cv2.IMREAD_COLOR)

        if self.is_visualise:
            print("showing image")
            # cv2.imshow("image_np1", image_np1)
            # cv2.imshow("image_np2", image_np2)
            # cv2.waitKey(1)
            visualise(image_np1, image_np2, "", wait=1)

        if self.is_record:
            if self.record_all_frames or self.counter % 10 == 0:
                cv2.imwrite(f"{self.recording_path}/LEFT/frame-{self.counter}.jpg", image_np1)
                cv2.imwrite(f"{self.recording_path}/RIGHT/frame-{self.counter}.jpg", image_np2)
        
        self.counter += 1

        time2 = timer()
        if (time2 - self.time1 > 1.0/20.0):
            print(f"[red]slow fps in callback(): {1.0/(time2 - self.time1)}" )

        self.time1 = time2

if __name__ == '__main__':
    ros_subscriber = Recorder()

    rospy.spin()
