from sensor_msgs.msg import CameraInfo
from utils.rosbag_to_dataset.dtypes.base import Dtype
import numpy as np

class CameraInfoConvert(Dtype):
    """
    Convert an odometry message into a 13d vec.
    """
    def __init__(self):
        pass

    def N(self):
        return 1

    def rosmsg_type(self):
        return CameraInfo

    def ros_to_numpy(self, msg):
        if msg is None:
            return None
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())
        return msg.k[np.newaxis, ...]

if __name__ == "__main__":
    c = Float64Convert()
    msg = Float64()

    print(c.ros_to_numpy(msg))
