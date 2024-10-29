# This file is to generate vnnlib files for the exported onnx models.

import os

def generate_vnnlib(input_list, perturb_list, output_range_low, output_range_high, dir_path):
    """
    Generates a VNNLIB file for the given ONNX model.

    Parameters:
    - model_path: Path to the PyTorch model checkpoint.
    - onnx_path: Path to the exported ONNX model.
    - vnnlib_path: Path where the VNNLIB file will be saved.
    """
    path = os.path.join(dir_path, "spec.vnnlib")
    with open(path, "w") as f:
        f.write("\n")
        for i in range(len(input_list)):    
            f.write(f"(declare-const X_{i} Real)\n")
        f.write(f"(declare-const Y_1 Real)\n")

        f.write("\n; Input constraints:\n")
        for i in range(len(input_list)):
            f.write(f"(assert (>= X_{i} {input_list[i] - perturb_list[i]}))\n")
            f.write(f"(assert (<= X_{i} {input_list[i] + perturb_list[i]}))\n")

        # negate
        f.write("\n; Output constraints:\n")
        if output_range_high is None:
            f.write(f"(assert (<= Y_1 {output_range_low}))\n")
        elif output_range_low is None:
            f.write(f"(assert (>= Y_1 {output_range_high}))\n")
        else:
            f.write(f"(assert (>= Y_1 {output_range_low}))\n")
            f.write(f"(assert (<= Y_1 {output_range_high}))\n")

    print(f"[DONE] Generated VNNLIB at {path}")
    
