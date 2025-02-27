import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from ml.models.SimpleNN import SimpleNN


def main() -> None:
    """Main function to visualize the SimpleNN model with TensorBoard."""

    # Define input data (example feature vector)
    x: list[float] = [
        142,20,13.147887323943662,142,0.0,7190,52.48175182481752,50211,363.8478260869565,0,2629,11455,145,3754,2252,322,2614,341,972,837,401,540,401,0.053042328042328,0.0714285714285714,0.053042328042328,24,1889,24,6,3,0,4,145,10,0,145,0.0206896551724137,0.0,0.0275862068965517,1.0,0.0689655172413793,0.0,1.0,258,217,17,175,1473,750,85,151,2,2,402
    ]

    # Convert input data to a PyTorch tensor and add batch dimension (1, 55)
    x_tensor: Tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    # Initialize the model
    model: SimpleNN = SimpleNN(len(x))

    # Create a TensorBoard writer
    writer: SummaryWriter = SummaryWriter(log_dir='runs/simple_nn_experiment')

    # Perform a forward pass through the model
    y: Tensor = model(x_tensor)

    # Add the model graph to TensorBoard
    writer.add_graph(model, x_tensor)

    # Close the TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()
