import torch
from torch import Tensor


class CudaHelper:
    """
    A utility class for handling CUDA operations in PyTorch.

    This class helps manage CUDA availability, device allocation, memory handling,
    and other CUDA-related operations to streamline GPU usage for efficient tensor
    computations. If CUDA is unavailable, operations default to CPU.

    Attributes:
        device (torch.device): Main device set to 'cuda' if available; otherwise, 'cpu'.
    """

    def __init__(self):
        """
        Initializes the CudaHelper with the primary device set to CUDA if available, or CPU otherwise.
        Displays a message about the device being used.
        """
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.is_cuda_available():
            print(f"CUDA is available. Using device: {self.device}")
        else:
            print("CUDA is not available. Using CPU.")

    @staticmethod
    def is_cuda_available() -> bool:
        """Checks if CUDA is available on the current system."""
        return torch.cuda.is_available()

    def get_device_name(self, device_id: int = 0) -> str:
        """Returns the name of the specified CUDA device, or a message if unavailable."""
        return torch.cuda.get_device_name(device_id) if self.is_cuda_available() else "No CUDA device available"

    @staticmethod
    def get_device_count() -> int:
        """Returns the number of available CUDA devices."""
        return torch.cuda.device_count()

    def list_all_cuda_devices(self) -> None:
        """Prints all CUDA device names along with their IDs."""
        if self.is_cuda_available():
            device_count: int = self.get_device_count()
            for device_id in range(device_count):
                device_name = torch.cuda.get_device_name(device_id)
                print(f"Device ID {device_id}: {device_name}")
        else:
            print("No CUDA devices available.")

    def select_device(self, device_id: int) -> None:
        """
        Selects a specific CUDA device by its ID. If the specified device ID is invalid or
        if CUDA is not available, defaults to CPU.

        Args:
            device_id (int): The ID of the CUDA device to select.
        """
        if self.is_cuda_available() and 0 <= device_id < self.get_device_count():
            self.device = torch.device(f"cuda:{device_id}")
            print(f"Selected device: {self.get_device_name(device_id)} (ID {device_id})")
        else:
            self.device = torch.device("cpu")
            print("Invalid device ID or CUDA unavailable. Defaulting to CPU.")

    def allocate_to_device(self, tensor: Tensor) -> Tensor:
        """Moves a tensor to the configured device (CUDA or CPU)."""
        return tensor.to(self.device)

    def free_memory(self) -> None:
        """Clears the GPU memory cache to free up unused memory if CUDA is available."""
        if self.is_cuda_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache.")

    def memory_info(self, device_id: int = 0) -> None:
        """Displays allocated, reserved, and free memory info for the specified CUDA device."""
        if self.is_cuda_available():
            allocated: float = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            reserved: float = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
            free: float = (reserved - allocated)
            print(f"Memory Allocated: {allocated:.2f} MB")
            print(f"Memory Reserved: {reserved:.2f} MB")
            print(f"Memory Free: {free:.2f} MB")

    def synchronize_device(self) -> None:
        """Synchronizes the CUDA device, ensuring all operations are complete."""
        if self.is_cuda_available():
            torch.cuda.synchronize()

    def set_manual_seed(self, seed: int) -> None:
        """Sets a manual seed for reproducibility across CPU and CUDA operations if available."""
        torch.manual_seed(seed)
        if self.is_cuda_available():
            torch.cuda.manual_seed_all(seed)
            print(f"Seed set to {seed} for reproducibility.")