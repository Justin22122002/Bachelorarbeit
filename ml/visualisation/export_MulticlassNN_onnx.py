import torch
from torch import Tensor
from ml.models.MulticlassNN import MulticlassNN


def main() -> None:
    """Main function to export the MulticlassNN model to ONNX format."""

    # Initialize the model
    model: MulticlassNN = MulticlassNN(55)

    # Set the model to evaluation mode (optional if only exporting the structure)
    model.eval()

    # Create example input data (dummy data)
    dummy_input: Tensor = torch.randn(1, 55)  # Batch size 1, 55 features

    # ONNX export parameters
    input_names: list[str] = ['input']
    output_names: list[str] = ['output_multi', 'output_binary']  # Two outputs: multiclass and binary

    # Export the model in ONNX format
    torch.onnx.export(
        model,
        (dummy_input,),
        '../saved_models/best_model_multiclassNN.onnx',
        input_names=input_names,
        output_names=output_names,
        verbose=True,
        opset_version=12
    )


if __name__ == '__main__':
    main()
