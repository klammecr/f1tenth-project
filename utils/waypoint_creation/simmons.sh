#! /bin/bash
python create_waypoints.py -m simmons_1.pgm -o output
python create_waypoints.py -m simmons_1.pgm -a output/annot_interp.txt -o output
python transform_waypoints_to_map.py -m simmons_1.pgm -w output/final_waypoints.txt -y simmons_1.yaml