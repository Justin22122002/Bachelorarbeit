### Erste Schritte:

Die Deep learning Ansätze nutzen alle Pytorch, teilweise werden auch teile von Tensorflow genutzt, weswegen tensorflow installiert sein sollte

### Installation
1. Optional: (Tensorflow wird nicht mehr in diesem Projekt) Installiere TensorFlow auf deinem System (beachte: verwende die richtige Python-Version, siehe https://www.tensorflow.org/install / https://www.tensorflow.org/install/pip):
   - Python 3.8 – 3.11 / 3.12
   - Windows 7 oder später (mit C++ Redistributable)
   - ```python3 -m pip install tensorflow```, wir verwenden die CPU-Version, da die CUDA-Version für die GPU Probleme verursacht. Weitere Infos: https://www.tensorflow.org/install/pip#windows-native
   - Überprüfe die Installation: ```python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"```
   ```
      --> tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable TF_ENABLE_ONEDNN_OPTS=0
      os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
      --> To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags
      --> To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
      import tensorflow as tf
      print("tensorflow version installed: " +  tf.__version__)
      print("CPU test: ")
      print(tf.reduce_sum(tf.random.normal([1000, 1000])))
      print("GPU test: ")
      print(tf.config.list_physical_devices('GPU'))
   ```
2. Installiere PyTorch:
- 5.1. CPU 
- ``` pip3/pip install torch torchvision torchaudio ```
- 5.2. CUDA
- Installiere die unterstützte CUDA-Version von https://pytorch.org/ von https://developer.nvidia.com/cuda-toolkit
- Installiere Visual Studio 2019 Community Edition
- Optional: Installiere eine kompatible Version von cuCNN von https://developer.nvidia.com/rdp/cudnn-archive
- ```pip3/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124```