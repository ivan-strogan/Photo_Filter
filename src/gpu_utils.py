"""GPU detection and configuration utilities."""

import torch
import logging
from typing import Tuple, Dict, Any

class GPUManager:
    """Manages GPU detection and configuration for accelerated processing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.gpu_info = None
        self._detect_gpu()

    def _detect_gpu(self) -> None:
        """Detect available GPU and set device."""
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

                self.gpu_info = {
                    'available': True,
                    'device_count': gpu_count,
                    'device_name': gpu_name,
                    'memory_gb': round(gpu_memory, 2),
                    'cuda_version': torch.version.cuda,
                    'device_type': 'CUDA'
                }

                self.logger.info(f"GPU detected: {gpu_name} with {gpu_memory:.2f}GB memory")

            elif torch.backends.mps.is_available():
                # Apple Silicon GPU support
                self.device = torch.device("mps")
                self.gpu_info = {
                    'available': True,
                    'device_count': 1,
                    'device_name': 'Apple Silicon GPU',
                    'memory_gb': 'Unified Memory',
                    'cuda_version': None,
                    'device_type': 'MPS'
                }

                self.logger.info("Apple Silicon GPU (MPS) detected")

            else:
                self.device = torch.device("cpu")
                self.gpu_info = {
                    'available': False,
                    'device_count': 0,
                    'device_name': 'CPU Only',
                    'memory_gb': 0,
                    'cuda_version': None,
                    'device_type': 'CPU'
                }

                self.logger.info("No GPU detected, using CPU")

        except Exception as e:
            self.logger.warning(f"Error detecting GPU: {e}")
            self.device = torch.device("cpu")
            self.gpu_info = {
                'available': False,
                'device_count': 0,
                'device_name': 'CPU Only (Error)',
                'memory_gb': 0,
                'cuda_version': None,
                'device_type': 'CPU'
            }

    def get_device(self) -> torch.device:
        """Get the configured device.

        Returns:
            PyTorch device object
        """
        return self.device

    def is_gpu_available(self) -> bool:
        """Check if GPU is available.

        Returns:
            True if GPU is available, False otherwise
        """
        return self.gpu_info['available'] if self.gpu_info else False

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information.

        Returns:
            Dictionary with GPU information
        """
        return self.gpu_info.copy() if self.gpu_info else {}

    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """Get optimal batch size based on available GPU memory.

        Args:
            base_batch_size: Base batch size for CPU processing

        Returns:
            Optimal batch size
        """
        if not self.is_gpu_available():
            return base_batch_size

        try:
            if self.gpu_info['device_type'] == 'CUDA':
                memory_gb = self.gpu_info['memory_gb']

                # Adjust batch size based on GPU memory
                if memory_gb >= 24:  # High-end GPU
                    return base_batch_size * 4
                elif memory_gb >= 12:  # Mid-range GPU
                    return base_batch_size * 2
                elif memory_gb >= 6:  # Entry-level GPU
                    return base_batch_size
                else:  # Low memory GPU
                    return max(base_batch_size // 2, 1)

            elif self.gpu_info['device_type'] == 'MPS':
                # Apple Silicon - conservative batch size
                return base_batch_size * 2

        except Exception as e:
            self.logger.warning(f"Error calculating optimal batch size: {e}")

        return base_batch_size

    def configure_model_for_device(self, model) -> None:
        """Configure a model for the detected device.

        Args:
            model: PyTorch model to configure
        """
        try:
            model.to(self.device)

            if self.is_gpu_available():
                # Enable optimizations
                if hasattr(model, 'eval'):
                    model.eval()

                # Enable inference mode optimizations
                if self.gpu_info['device_type'] == 'CUDA':
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False

                self.logger.info(f"Model configured for {self.gpu_info['device_type']} device")
            else:
                self.logger.info("Model configured for CPU device")

        except Exception as e:
            self.logger.error(f"Error configuring model for device: {e}")

    def clear_gpu_cache(self) -> None:
        """Clear GPU cache to free memory."""
        try:
            if self.gpu_info['device_type'] == 'CUDA':
                torch.cuda.empty_cache()
                self.logger.debug("CUDA cache cleared")
            elif self.gpu_info['device_type'] == 'MPS':
                torch.mps.empty_cache()
                self.logger.debug("MPS cache cleared")
        except Exception as e:
            self.logger.warning(f"Error clearing GPU cache: {e}")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage.

        Returns:
            Dictionary with memory usage information
        """
        memory_info = {
            'allocated_gb': 0,
            'cached_gb': 0,
            'total_gb': 0,
            'free_gb': 0
        }

        try:
            if self.gpu_info['device_type'] == 'CUDA':
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                cached = torch.cuda.memory_reserved(0) / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                memory_info.update({
                    'allocated_gb': round(allocated, 2),
                    'cached_gb': round(cached, 2),
                    'total_gb': round(total, 2),
                    'free_gb': round(total - cached, 2)
                })

        except Exception as e:
            self.logger.warning(f"Error getting memory usage: {e}")

        return memory_info

    def optimize_for_inference(self) -> None:
        """Apply optimizations for inference workloads."""
        try:
            if self.is_gpu_available():
                # Disable gradient computation globally
                torch.set_grad_enabled(False)

                if self.gpu_info['device_type'] == 'CUDA':
                    # Enable optimizations for inference
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False

                    # Use mixed precision if available
                    if hasattr(torch.cuda, 'amp'):
                        self.logger.info("CUDA optimizations enabled for inference")

                elif self.gpu_info['device_type'] == 'MPS':
                    self.logger.info("MPS optimizations enabled for inference")

        except Exception as e:
            self.logger.warning(f"Error applying inference optimizations: {e}")

    def print_gpu_summary(self) -> None:
        """Print a summary of GPU configuration."""
        if self.gpu_info:
            print(f"\n{'='*50}")
            print("GPU Configuration Summary")
            print(f"{'='*50}")
            print(f"Device Type: {self.gpu_info['device_type']}")
            print(f"Available: {self.gpu_info['available']}")

            if self.gpu_info['available']:
                print(f"Device Name: {self.gpu_info['device_name']}")
                print(f"Device Count: {self.gpu_info['device_count']}")

                if isinstance(self.gpu_info['memory_gb'], (int, float)):
                    print(f"Memory: {self.gpu_info['memory_gb']} GB")
                else:
                    print(f"Memory: {self.gpu_info['memory_gb']}")

                if self.gpu_info['cuda_version']:
                    print(f"CUDA Version: {self.gpu_info['cuda_version']}")

                print(f"PyTorch Device: {self.device}")

            print(f"{'='*50}\n")


# Global GPU manager instance
gpu_manager = GPUManager()