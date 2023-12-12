#! /bin/bash
source .venv/bin/activate
python utils/rosbag_to_dataset/converter/converter.py --bag_dir /media/cklammer/KlammerData1/data/f1tenth --seq 20231210_173215 --output_dir output
python utils/rosbag_to_dataset/converter/converter.py --bag_dir /media/cklammer/KlammerData1/data/f1tenth --seq 20231210_173840 --output_dir output
python utils/rosbag_to_dataset/converter/converter.py --bag_dir /media/cklammer/KlammerData1/data/f1tenth --seq 20231210_174401 --output_dir output
python utils/rosbag_to_dataset/converter/converter.py --bag_dir /media/cklammer/KlammerData1/data/f1tenth --seq 20231210_183138 --output_dir output