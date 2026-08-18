
## `README.md`



````markdown
# AI-Based Restoration of Degraded Semiconductor Images

## Overview

This repository contains our solution for the KLA Problem Statement:
**AI-Based Restoration of Degraded Images**.

The objective is to restore degraded grayscale semiconductor microscopy images while recovering the target spatial resolution.

The model takes a degraded low-resolution image as input and produces a restored image at the target resolution.

---

## Repository Structure

```text
casual/
│
├── murali/
│   ├── run.py
│   ├── requirements.txt
│   │
│   └── models/
│       └── best_model.pth
│
├── images/
│   └── output_image
│
├── raw_experiments/
│   ├── biomemeicsemiconindia.ipynb
│   ├── Other_models.txt
│   └── semiconindia.ipynb
│
├── Test Outputs(drive-link).txt
├── README.md
├── requirements.txt
└── training_script colab.ipynb
````

### Important files

* `run.py`
  Main inference entry point.

* `requirements.txt`
  Python dependencies required to run the solution.

* `models/best_model.pth`
  Trained restoration model weights.

* `training/training_script_colab.ipynb`
  Training procedure used to reproduce the model.

---

# 1. Requirements

## Hardware

The solution is designed to run on an NVIDIA GPU.

Recommended:

* NVIDIA GPU
* CUDA-compatible PyTorch installation
* At least 8 GB GPU memory

The solution automatically uses CUDA when an NVIDIA GPU is available.

If CUDA is unavailable, the code can fall back to CPU, although GPU inference is recommended.

---

# 2. Software

Recommended environment:

* Python 3.10+
* PyTorch
* NumPy
* scikit-image

All required Python packages are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

Do NOT use:

```bash
pip install requirements.txt
```

The `-r` flag is required because the file contains a list of packages.

---

# 3. Model Weights

The trained model weights are included locally in:

```text
murali/models/best_model.pth
```

The inference script loads the weights directly from this local file.

---

# 4. Input Format

The inference script expects a directory containing degraded images in NumPy format:

```text
.npy
```

Each input file should contain a grayscale image array.

Example:

```text
input/
├── 000000.npy
├── 000001.npy
├── 000002.npy
└── ...
```

The input arrays are expected to represent the degraded/low-resolution images used by the model.

For the supplied dataset, the degraded images have spatial dimensions:

```text
128 × 128
```

---

# 5. Running Inference

The required command is:

```bash
python run.py <input-dir> <output-dir>
```

For example:

```bash
python run.py ./input ./output
```

If the output directory does not already exist, `run.py` creates it automatically.

---

# 6. Output Format

The script generates one `.npy` file for every input `.npy` file.

For example:

```text
input/
├── 000000.npy
├── 000001.npy
└── 000002.npy
```

produces:

```text
output/
├── 000000.npy
├── 000001.npy
└── 000002.npy
```

The output filenames are identical to the corresponding input filenames.

The restored output is:

* grayscale
* NumPy `.npy` format
* target spatial resolution
* finite-valued
* clipped to the range `[0, 1]`

For the supplied dataset, the expected restored resolution is:

```text
256 × 256
```

---

# 7. Example

Create an input directory:

```bash
mkdir input
mkdir output
```

Place the degraded `.npy` files inside `input/`.

Then execute:

```bash
python run.py input output
```

After inference:

```text
output/
├── 000000.npy
├── 000001.npy
├── 000002.npy
└── ...
```

---

# 8. GPU Execution

The inference script automatically detects CUDA.

If an NVIDIA GPU is available:

```text
CUDA
```

is used automatically.

Otherwise:

```text
CPU
```

is used as a fallback.

No manual GPU configuration is required.

The solution does not require:

* internet access
* API keys
* Hugging Face authentication
* external model downloads
* user interaction
* manual configuration

---

---

# 9. Model Architecture

The restoration network is a convolutional encoder-decoder architecture based on a U-Net-style structure.

The network contains:

* encoder blocks
* downsampling layers
* bottleneck layers
* transposed-convolution upsampling
* decoder blocks
* skip connections
* residual reconstruction

The final prediction combines the reconstructed residual with an interpolated input baseline.

The architecture is fully convolutional and operates directly on the degraded grayscale image.

---

# 10. Training

The training procedure used to obtain the submitted weights is provided in:

```text
training/training_script_colab.ipynb
```

The training process uses paired degraded images and corresponding ground-truth images during the supervised training stages.

The final model was further refined using additional restoration objectives designed to improve:

* pixel fidelity
* structural similarity
* edge preservation
* local contrast
* multi-scale structural information

The training notebook contains the implementation used to generate the trained model.

---

# 11. Reproducing Training

To reproduce training, open:

```text
training/training_script_colab.ipynb
```

The notebook contains the training implementation and model definition.

Training requires the original training dataset and an NVIDIA GPU.

Training is computationally more expensive than inference.

The submitted inference weights are provided separately in:

```text
models/best_model.pth
```

Therefore, reviewers do not need to retrain the model to run inference.

---

# 12. Verification

After running inference, the following properties can be checked for every output:

```python
import numpy as np

x = np.load("output/000000.npy")

print("Shape:", x.shape)
print("Min:", x.min())
print("Max:", x.max())
print("Finite:", np.isfinite(x).all())
```

Expected output properties:

```text
Shape: (256, 256)
Min: >= 0
Max: <= 1
Finite: True
```

---

# 13. Reproducibility

The inference pipeline is deterministic with respect to the model weights and input data.

All model weights required for inference are included locally.

The inference process does not communicate with external services.

No API keys, network services, or online model repositories are required.

---

# 14. Quick Start

For a reviewer, the complete process is:

### Step 1 — Clone the repository

```bash
git clone https://github.com/deep6511/casual.git
cd casual\murali
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Prepare input data

Place the degraded `.npy` files in:

```text
input/
```

### Step 4 — Run inference

```bash
python run.py input output
```

### Step 5 — Check results

Restored images will be available in:

```text
output/
```

with the same filenames as the input images.

---

## Expected Command

The complete inference interface is:

```bash
python run.py <input-dir> <output-dir>
