# This script is to create a new benchmark from scratch
from tools.export import export_to_onnx
from tools.generate_vnnlib import generate_vnnlib
from tools.verify_abcrown import run_verification

import yaml
import importlib.util
import torch
import argparse
import os

def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(model_path, model_python_path, model_name):
    """Load the specified model class without executing the script."""

    # Load the model class from the specified Python file
    spec = importlib.util.spec_from_file_location(model_name, model_python_path)
    model_module = importlib.util.module_from_spec(spec)

    # Load the module without executing its code
    loader = spec.loader
    if loader:
        loader.exec_module(model_module)

    # Retrieve the model class from the loaded module
    model_class = getattr(model_module, model_name, None)

    if model_class:
        # Instantiate the model and load the checkpoint
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set model to evaluation mode
        return model
    else:
        raise ImportError(f"Model class {model_name} not found in {model_python_path}.")

def get_directory(file_path):
    """Returns the directory of the given file path using string slicing."""
    last_slash_index = file_path.rfind('/')
    if last_slash_index == -1:
        return ''  # Return an empty string if there's no directory in the path
    return file_path[:last_slash_index]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load model with configuration file")
    parser.add_argument(
        "--config",
        default="configuration.yaml",
        type=str,
        required=True,
        help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Load the configuration file
    config = load_config(args.config)

    # Extract paths from the configuration
    pt_path = config["paths"]["pt_path"]
    model_architecture_path = config["paths"]["model_architecture_path"]
    model_name = config["paths"]["model_name"]

    # Load the model
    model = load_model(pt_path, model_architecture_path, model_name)
    print("Model loaded successfully.")

    # export onnx
    # Get the directory of the model file
    directory = get_directory(pt_path)
    onnx_path = os.path.join(directory, "model.onnx")
    export_to_onnx(model, onnx_path)

    # generate instance
    input_list = config['inputs']['input_list']
    perturbation_list = config['inputs']['perturbation_list']
    generate_vnnlib(input_list, perturbation_list, config['outputs']['output_lower'], config['outputs']['output_upper'], directory)

    # run verification
    abcrown_path = config['paths']['abcrown_path']
    run_verification(onnx_path, config['paths']['vnnlib_path'], abcrown_path)
    

if __name__ == "__main__":
    main()
