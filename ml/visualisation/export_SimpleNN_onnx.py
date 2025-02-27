import torch
from torch import Tensor
from ml.models.SimpleNN import SimpleNN


def main() -> None:
    """Main function to export the SimpleNN model to ONNX format."""

    # Initialize the model
    model: SimpleNN = SimpleNN(55)

    # Create example input data (dummy data)
    dummy_input: Tensor = torch.randn(1, 55)  # Batch size 1, 55 features

    # ONNX export parameters
    input_names: list[str] = ['input']
    output_names: list[str] = ['output']

    # If the model expects only one input, wrap it in a tuple
    torch.onnx.export(
        model,
        (dummy_input,),
        '../saved_models/best_model_simpleNN.onnx',
        input_names=input_names,
        output_names=output_names
    )


if __name__ == '__main__':
    main()
