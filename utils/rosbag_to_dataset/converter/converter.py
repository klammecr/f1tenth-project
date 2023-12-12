# Third Party
import os
import numpy as np
import argparse
from pathlib import Path
from rosbags.highlevel import AnyReader
import cv2

# ROS Imports
import rospy
from cv_bridge import CvBridge

# In House
from utils.rosbag_to_dataset.dtypes.float64 import Float64Convert
from utils.rosbag_to_dataset.dtypes.odometry import OdometryConvert
from utils.rosbag_to_dataset.dtypes.image import ImageConvert
from utils.rosbag_to_dataset.dtypes.ackermann_drive import AckermannDriveConvert
from utils.rosbag_to_dataset.dtypes.twist import TwistConvert
from utils.rosbag_to_dataset.dtypes.imu import ImuConvert
from utils.rosbag_to_dataset.dtypes.pose import PoseConvert
from utils.rosbag_to_dataset.dtypes.gridmap import GridMapConvert
from utils.rosbag_to_dataset.dtypes.pointcloud import PointCloud2Convert

# Data type converters
dtype_convert = {
        "Float64":Float64Convert,
        "Odometry":OdometryConvert,
        "Image":ImageConvert,
        "AckermannDrive":AckermannDriveConvert,
        "Twist":TwistConvert,
        "Imu":ImuConvert,
        "Pose":PoseConvert,
        "GridMap":GridMapConvert,
        "PointCloud2":PointCloud2Convert
    }

# This code is derived from the TartanDrive github repo:
# https://github.com/castacks/tartan_drive/blob/main/rosbag_to_dataset/rosbag_to_dataset/converter/converter.py

class Converter:
    """
    Rosnode that converts to numpy using the format specified by the config spec.
    Current way I'm handling this is by saving all the messages necessary, then processing them all at once.
    This way there shouldn't be any latency problems (as not all fields will be timestamped)

    Ok, the best paradigm for advancing time here is to have a single timer.
    Once a topic has added to its buf for that timestep, set a flag.
    Once dt has passed, reset the flags.
    We can error check by looking at flags.
    Real-time should be ok-ish as we don't process anything. If not, we can use stamps for the topics that have those.
    """
    def __init__(self, use_stamps, topic_map, dt = None):
        """
        Args:
            spec: As provided by the ConfigParser module, the sizes of the observation/action fields.
            converters: As provided by the ConfigParser module, the converters from ros to numpy.
        """
        self.queue = {}
        self.dt = dt
        self.use_stamps = use_stamps
        self.topic_map = topic_map

    def reset_queue(self):
        """
        Reset the queue with empty dicts for each topic.
        """
        for topic, remap in self.topic_map.items():
            self.queue[topic] = []

    def preprocess_queue(self, rates):
        """
        Do some smart things to fill in missing data if necessary.
        """
        """
        import matplotlib.pyplot as plt
        for k in self.queue.keys():
            data = [0. if x is None else 1. for x in self.queue[k]]
            plt.plot(data, label=k)
        plt.legend()
        plt.show()
        """
        #Start the dataset at the point where all topics become available
        print('Preprocessing...')
        data_exists = {}
        strides = {}
        start_idxs = {}
        for k in self.queue.keys():
            data_exists[k] = [not x is None for x in self.queue[k]]
            strides[k] = int(self.dt/rates[k])
            start_idxs[k] = data_exists[k].index(True) // strides[k]

        #This trick doesn't work with differing dts
        #thankfully, index gives first occurrence of value.
        start_idx = max(start_idxs.values())

        #For now, just search backward to fill in missing data.
        #This is most similar to the online case, where you'll have stale data if the sensor doesn't return.
        for k in self.queue.keys():
            last_avail = start_idx * strides[k]
            for t in range(start_idx, len(self.queue[k])):
                if data_exists[k][t]:
                    last_avail = t
                else:
                    self.queue[k][t] = self.queue[k][last_avail]

        self.queue = {k:v[start_idx*strides[k]:] for k,v in self.queue.items()}

    def calculate_rates(self):
        """Calculate rates for each topic of interest

        Args:
            bag (rospy.Bag): The deserialized bag file
            topics (list): The topics of interest
        """
        rates = {}
        for k, v in self.topic_map.items():
            # self.dt      = info_dict["duration"]
            # rates[topic] = self.dt / bag.get_message_count(topic)
            rates[k] = self.dt
        return rates

    def convert_queue(self, rates, data_types):
        """
        Convert the queue to a numpy array.
        """
        out = {}
        strides = {k:int(self.dt/rates[k]) for k in self.queue.keys()}
        for topic, remap in self.topic_map.items():
            data_type = data_types[topic]
            trim_type = data_type.split("/")[-1]
            if "Pose" in trim_type:
                trim_type = "Pose"
            data      = self.queue[topic]
            try:
                # Create the converter
                if trim_type == "Image":
                    if "color" in topic:
                        args = (3, (1024, 544))
                    else:
                        # Original resolution according to spec
                        args = (1, (1024, 544))
                        # Power of 2
                        # args = (1, (1024, 512))
                elif trim_type == "Odometry":
                    args = (False, False)
                elif trim_type == "GridMap":
                    args = (3, (501,501))
                elif trim_type == "PointCloud2":
                    # Size of the disparity iamge was 512x256
                    args = ((512,256), True)
                else:
                    args = ()

                # Data type converter!
                converter = dtype_convert[trim_type](*args)
                
            except:
                raise ValueError(f"Cannot find converter for {topic} please add one.")
    
            # Convert the message to numpy arrays
            conv_data = [part for x in data if np.all(part := converter.ros_to_numpy(x)) != None]
            np_data   = np.stack(conv_data, axis = 0)

            if strides[topic] > 1:
                #If strided, need to reshape data (we also need to (same) pad the end)
                pad_t   = strides[topic] - (np_data.shape[0] % strides[topic])
                np_data = np.concatenate([np_data, np.stack([np_data[-1]] * pad_t, axis=0)], axis=0)
                np_data = data.reshape(-1, strides[topic], *data.shape[1:])

            # Put in the dict
            out[remap] = np_data
        return out

    def preprocess_pose(self, traj, zero_init=False):
        """
        Sliding window to smooth out the pose.

        Args:
            traj (_type_): _description_
            zero_init (bool, optional): _description_. Defaults to False.
        """
        N = 2
        T = traj["odom"].shape[0]
        pad_poses    = torch.cat([traj["odom"][[0]]] * N + [traj["odom"]] + [traj["odom"][[-1]]] * N)
        smooth_poses = torch.stack([pad_poses[i:T+i] for i in range(N*2 + 1)], dim=-1).mean(dim=-1)[:, :3]

        # Insert the smoothed poses
        traj["odom"][:, :3] = smooth_poses

        return traj

    def traj_to_torch(self, traj):
        """
        Save the trajectory of the vehicles to Pytorch files

        Args:
            traj (dict): Dictionary of keys for topics and values for the data
        """
        torch_traj = {}
        # Make uniform size
        all_size = np.array([traj[key].shape[0] for key in traj.keys()])
        traj_len = np.min(all_size)
        # Enable this to force everything to be the same length
        # traj = {k: v[:traj_len] for k,v in traj.items()}
        
        # Check for nan values
        max_nan_idx = -1
        for i in range(traj_len):
            nan = [np.any(traj[k] is np.inf or traj[k] is np.nan) for k in traj.keys()]
            if np.any(nan):
                max_nan_idx = i

        start_idx = max_nan_idx + 1
        for k, v in traj.items():
            torch_traj[k] = torch.tensor(v[start_idx:]).float()

        return torch_traj

    def convert_bag(self, bag_file, as_torch=False, zero_pose_init=False):
        """
        Convert a bag into a dataset.
        """
        # Setup
        print('Extracting Messages...')
        self.reset_queue()

        # Iterate through the ROS1 or ROS2 bag
        topic_curr_idx = {k:0 for k in self.topic_map.keys()}
        with AnyReader([Path(bag_file)]) as reader:
            # Setup Data types
            all_topics = reader.topics
            data_types = {topic: all_topics[topic].msgtype for topic in all_topics.keys() if topic in self.topic_map.keys()}
            for k in self.topic_map.keys():
                assert k in all_topics, "Could not find topic {} from envspec in the list of topics for this bag.".format(k)

            # Calculate timesteps
            rates = self.calculate_rates()
            timesteps = {k:np.arange(reader.start_time*1e-9, reader.end_time*1e-9, rates[k]) for k in self.topic_map.keys()}

            for connection, ts, data in reader.messages():
                # See if we can deserialize
                try:
                    msg = reader.deserialize(data, connection.msgtype)
                except:
                    print(f"Could not deserialize: {connection.msgtype}")
                    continue
            
                if connection.topic in self.topic_map.keys():
                    tidx = topic_curr_idx[connection.topic]

                    #Check if there is a stamp and it has been set.
                    has_stamp = msg.header.stamp.sec > 1000.
                    has_info  = msg.header.stamp.sec > 1000.

                    #Use the timestamp if its valid. Otherwise default to rosbag time.
                    if (has_stamp or has_info) and self.use_stamps:
                        stamp = msg.header.stamp if has_stamp else msg.info.header.stamp
                        topic = connection.topic
                        stamp_float = float(f"{stamp.sec}.{stamp.nanosec}")
                        if (tidx < timesteps[topic].shape[0]) and stamp_float > timesteps[topic][tidx]:
                            #Add to data. Find the smallest timestep that's less than t.
                            idx = np.searchsorted(timesteps[topic], stamp_float)
                            topic_curr_idx[topic] = idx

                            #In case of missing data.
                            while len(self.queue[topic]) < idx:
                                self.queue[topic].append(None)

                            self.queue[topic].append(msg)
                    else:
                        if (tidx < timesteps[topic].shape[0]) and (t > rospy.Time.from_sec(timesteps[topic][tidx])):
                            #Add to data. Find the smallest timestep that's less than t.
                            idx = np.searchsorted(timesteps[topic], t.to_sec())
                            topic_curr_idx[topic] = idx

                            #In case of missing data.
                            while len(self.queue[topic]) < idx:
                                self.queue[topic].append(None)

                            self.queue[topic].append(msg)

        # Convert to dataset
        self.preprocess_queue(rates)
        res = self.convert_queue(rates, data_types)
        if as_torch:
            torch_traj = self.traj_to_torch(res)
            torch_traj = self.preprocess_pose(torch_traj, zero_pose_init)
            return torch_traj
        else:
            return res

def get_bagfile_name_no_ext(filepath):
    filename = filepath.split("/")[-1]
    filename_no_ext = filename.split(".")[0]
    return filename_no_ext

def convert_dir_of_bags(conv : Converter, bag_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for file in os.listdir(bag_dir):
        filepath = f"{bag_dir}/{file}"
        filename_no_ext = get_bagfile_name_no_ext(filepath)
        if os.path.isfile(filepath) and filepath.endswith("bag"):
            bag = rosbag.Bag(filepath, "r")
            out = conv.convert_bag(bag, as_torch=True, zero_pose_init=False)
            torch.save(out, f"{out_dir}/{filename_no_ext}.pth")

if __name__ == "__main__":
    """
    Parse TartanDrive bag files and save to HDF5 to be loaded as a torch dataset later.
    """
    parser = argparse.ArgumentParser(description="Tartan Drive Bag Parser")
    parser.add_argument("--bag_dir", help="Input ROS bag.", default="/media/cklammer/KlammerData1/data/f1tenth")
    parser.add_argument("--seq", help="Input ROS bag.", default="20231210_173840")
    parser.add_argument("--output_dir", help="Output directory.", default="output")
    args = parser.parse_args()
    
    # Setup paths
    bag_loc = f"{args.bag_dir}/{args.seq}"
    out_loc = f"{args.output_dir}/{args.seq}"
    if not os.path.exists(out_loc):
        os.mkdir(out_loc)
    
    # Instaniate Converter
    topic_map = {
                   "/pf/viz/inferred_pose":"odom", # Raw odometry from VIO, this is from the world frame, we want this for GPS
                   "/depth/image_rect_raw": "depth_img",
                   "/color/image_raw": "color_img",
                #   "/color/camera_info" : None,
                #    "/color/metadata" : None,
                #    "/depth/camera_info" : None,
                #    "/depth/metadata" : None,
                #    "/extrinsics/depth_to_color" : None,
                #    "/extrinsics/depth_to_infra1" : None,
                #    "/extrinsics/depth_to_infra2" : None
                }
    conv = Converter(True, topic_map=topic_map, dt=0.1)

    # Single Bag File
    out = conv.convert_bag(bag_loc, as_torch=False, zero_pose_init=False)

    for k, v in out.items():
        topic_dir = f"{out_loc}/{k}"
        if not os.path.exists(topic_dir):
            os.mkdir(topic_dir)
        
        if "img" in k:
            for i in range(v.shape[0]):
                img = v[i]
                img_swp = np.transpose(img, (1, 2, 0))
                img_swp = cv2.cvtColor(img_swp, cv2.COLOR_RGB2BGR)
                if img_swp.shape[-1] == 1:
                    img_swp = img_swp[:, :, 0]
                cv2.imwrite(f"{topic_dir}/{i}.png", (img_swp*255).astype("uint8"))
        else:
            np.savetxt(f"{topic_dir}/{k}.npy", v)