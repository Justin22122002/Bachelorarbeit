import torch

from ml.cuda.CudaHelper import CudaHelper


def main() -> None:
    # Create an instance of the CudaHelper class
    cuda_helper = CudaHelper()

    # Check if CUDA is available
    print("CUDA available:", cuda_helper.is_cuda_available())

    # Get the name of the CUDA device (if available)
    print("CUDA device name:", cuda_helper.get_device_name())

    # Get the number of CUDA devices
    print("Number of CUDA devices:", cuda_helper.get_device_count())

    # Create a sample tensor and move it to the CUDA device
    tensor: torch.Tensor = torch.rand(3, 3)
    tensor_on_device: torch.Tensor = cuda_helper.allocate_to_device(tensor)
    print("Tensor moved to device:", tensor_on_device)

    cuda_helper.list_all_cuda_devices()

    # Display memory information for the device
    cuda_helper.memory_info()

    # Clear CUDA memory cache
    cuda_helper.free_memory()

    # Synchronize the CUDA device to ensure all operations are complete
    cuda_helper.synchronize_device()

    # Set a seed for reproducible results
    cuda_helper.set_manual_seed(42)

if __name__ == "__main__":
    main()
