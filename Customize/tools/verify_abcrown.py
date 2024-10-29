# This file is to use abcrown to verify the instances
'''
This script is used to verify models using abcrown
'''

import argparse
import subprocess

def get_directory(file_path):
    """Returns the directory of the given file path using string slicing."""
    last_slash_index = file_path.rfind('/')
    if last_slash_index == -1:
        return ''  # Return an empty string if there's no directory in the path
    return file_path[:last_slash_index]

def create_yaml(yaml_path: str, vnnlib_path: str, onnx_path: str) -> None:
    """
    Creates a YAML configuration file for ABCrown with the specified model and specification paths.

    Args:
        yaml_path (str): The path to save the YAML configuration file.
        vnnlib_path (str): Path to the VNN-LIB specification file.
        onnx_path (str): Path to the ONNX model file.
    """
    config_content = (
        "general:\n"
        "  enable_incomplete_verification: False\n"
        "  conv_mode: matrix\n"
        "model:\n"
        f"  onnx_path: {onnx_path}\n"
        "specification:\n"
        f"  vnnlib_path: {vnnlib_path}\n"
        "solver:\n"
        "  batch_size: 2048\n"
        "bab:\n"
        "  branching:\n"
        "    method: sb\n"
        "    sb_coeff_thresh: 0.1\n"
        "    input_split:\n"
        "      enable: True\n"
    )
    with open(yaml_path, 'w') as f:
        f.write(config_content)

def run_verification(onnx_path: str, vnnlib_path: str, abcrown_path: str, output_file: str = None, yaml_path: str = None) -> None:
    """
    Runs the ABCrown verification process on the given ONNX model and VNN-LIB specification.

    Args:
        onnx_path (str): Path to the ONNX model file.
        vnnlib_path (str): Path to the VNN-LIB specification file.
        abcrown_path (str): Path to the ABCrown script.
        output_file (str, optional): Path to save the verification output. Defaults to "output.txt" in model's directory.
        yaml_path (str, optional): Path for the YAML config file. Defaults to "model.yaml" in model's directory.
    """
    if yaml_path is None:
        yaml_path = os.path.join(get_directory(onnx_path), "model.yaml")
    create_yaml(yaml_path, vnnlib_path, onnx_path)

    if output_file is None:
        output_file = os.path.join(get_directory(onnx_path), "output.txt")

    with open(output_file, 'w') as f:
        subprocess.run(['python', abcrown_path, '--config', yaml_path], stdout=f, stderr=subprocess.STDOUT)

    print(f"[Done] Finished running ABCrown verification. Results saved in {output_file}")