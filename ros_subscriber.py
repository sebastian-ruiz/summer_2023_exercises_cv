# https://github.com/rospypi/simple
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

class ROSSubscriber():
    def __init__(self) -> None:
        rospy.init_node("client")
        rospy.loginfo('start')

        self.time1 = timer()

        self.queue = Queue.LifoQueue() # thread safe

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

        # cv2.imshow("image_np1", image_np1)
        # cv2.imshow("image_np2", image_np2)
        # cv2.waitKey(1)
        self.queue.put((image_np1, image_np2))

        time2 = timer()
        if (time2 - self.time1 > 1.0/20.0):
            print(f"[red]slow fps in callback(): {1.0/(time2 - self.time1)}" )

        self.time1 = time2

    def read(self):
        if self.queue.qsize():
            image_np1, image_np2 = self.queue.get()
            with self.queue.mutex:
                self.queue.queue.clear()
            # return ret, cv2.flip(frame, -1), cam_fps, img_created
            return image_np1, image_np2

        else:
            return None, None

if __name__ == '__main__':
    ros_subscriber = ROSSubscriber()

    time1 = timer()
    
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        image_np1, image_np2 = ros_subscriber.read()
        if image_np1 is not None:
            # print("showing img")
            visualise(image_np1, image_np2, text=None, wait=1)

            time2 = timer()
            if (time2 - time1 > 1.0/25.0):
                print(f"[red]slow fps in read(): {1.0/(time2 - time1)}")

            time1 = time2

        rate.sleep()

    # todo: use with to do rospy.is_shutdown
