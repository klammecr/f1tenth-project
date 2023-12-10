#! /bin/bash
python create_waypoints.py -m atrium.pgm -o output
python create_waypoints.py -m atrium.pgn -a output/annot_interp.txt -o output
python transform_waypoints_to_map.py -m atrium.pgm -w output/final_waypoints.txt -y atrium.yaml