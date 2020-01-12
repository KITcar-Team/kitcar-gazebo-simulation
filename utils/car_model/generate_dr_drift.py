#!/usr/bin/env python3

import yaml
import io
import argparse, sys,os

import vehicle_model as vehicle_model
import camera_calibration as calibration
import numpy as np



def read_yaml(input_file):
    with open(input_file) as stream:
        data = yaml.load(stream)
    return data


def generate(model_base_path, input_yaml):

    data = read_yaml(input_yaml)

    #Do camera calibration
    cam_data = data['front_camera'] #Extract data
    cam_calibration_yaml = calibration.create_camera_yaml(cam_data) #create calibration yaml
    cam_horizontal_fov = calibration.fov(f= cam_data['focal_length'], res = cam_data['capture']['width'])#Calculate horizontal fov

#Do camera calibration
    #depth_cam_data = data['depth_camera'] #Extract data
    #depth_cam_calibration_yaml = calibration.create_camera_yaml(depth_cam_data) #create calibration yaml
    #depth_cam_horizontal_fov = calibration.fov(f= depth_cam_data['focal_length'], res = depth_cam_data['capture']['width'])#Calculate horizontal fov
    depth_cam_horizontal_fov = None
    depth_cam_calibration_yaml = None

    model_xml = vehicle_model.extend_dr_drift(model_base_path, data, cam_horizontal_fov, depth_cam_horizontal_fov)

    return cam_calibration_yaml, depth_cam_calibration_yaml, model_xml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate calibration file and model file")
    parser.add_argument("input", nargs="?")
    parser.add_argument("--model_base_path", "-b", required=True)
    parser.add_argument("--calibration_output", "-c", required=True)
    parser.add_argument("--depth_calibration_output", "-d", required=True)
    parser.add_argument("--model_out_path", "-m", required=True)
    args = parser.parse_args()

    cal, depth_cal, model = generate(args.model_base_path, input_yaml = args.input)

    # Write to files

    with open(args.calibration_output,'w+') as file:
        file.write(cal)

#    with open(args.depth_calibration_output,'w+') as file:
#        file.write(depth_cal)
    
    with open(args.model_out_path,'w+') as file:
        file.write(model)

