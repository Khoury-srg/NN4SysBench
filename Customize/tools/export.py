# This file is to export a pytorch model to onnx format.
import torch

def export_to_onnx(model, export_path, input_size=(1, 2)):
    """
    Exports the given PyTorch model to ONNX format.

    Parameters:
    - model: The PyTorch model to be exported.
    - export_path: Path where the ONNX model will be saved.
    - input_size: Tuple representing the size of the model's input.
                  Default is (1, 2) for a model with 2 input features.
    """
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor with the specified input size
    dummy_input = torch.randn(*input_size)

    # Export the model
    torch.onnx.export(
        model, 
        dummy_input, 
        export_path, 
        export_params=True,       # Store the trained parameter weights inside the ONNX model
        opset_version=11,         # Set the ONNX opset version
        do_constant_folding=True, # Optimize constant folding for operations
        input_names=['input'],    # Specify the input layer name
        output_names=['output']   # Specify the output layer name
    )
    print(f"Model has been exported to {export_path}")

# Example usage
# Assuming `model` is the PyTorch model instance and you want to save it to "model.onnx"
# export_to_onnx(model, "model.onnx")
