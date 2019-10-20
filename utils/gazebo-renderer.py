#!/usr/bin/env python3
from road_generation.renderer import sdf
import argparse, sys, os

SEGMENTATION = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Gazebo SDF files from CommonRoad XML")
    parser.add_argument("input", nargs="?", type=argparse.FileType("r"),
        default=sys.stdin)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--name", "-n", required=True)
    parser.add_argument("--generators","-g", required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--lines","-l", action="store_true")
    parser.add_argument("--background","-bg", action="store_true")
    parser.add_argument("--segmentation","-s", action="store_true")
    parser.add_argument("--tile_size", "-t")
    #parser.add_argument("--add_vehicle", "-av", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if os.listdir(args.output) != [] and not args.force:
        print("Output directory is not empty.")
        print("Use --force")
        sys.exit(1)

    with args.input as input_file:
        xml = input_file.read()

    generate_args = {
        'xml_content':xml, #Content of commonroad
        'target_dir':args.output, #Directory which will contain the world.sdf file
        'generator_dir':args.generators, #Directory which contains commonroad
        'road_name':args.name, #Name of road
        'add_vehicle':True, #Add dr_drift model to world
        'background':args.background, #If background wall images should be included
        'segmentation':args.segmentation #If segmentation colors should be used for rendering
    }


    #Check if tiles should be extracted
    extract_tiles = not (args.tile_size is None)
    tile_size = None
    if extract_tiles:
        generate_args['extract_tiles'] = True
        generate_args['tile_size'] = float(args.tile_size)


    sdf.generate_sdf(**generate_args)
