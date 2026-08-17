import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# ============================================================

class RestorationNet(nn.Module):

    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )

        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )

        self.dec1 = conv_block(128, 64)

        self.final_up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        self.final_conv = nn.Conv2d(
            64, 1, kernel_size=1
        )

    def forward(self, x):

        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        b = self.bottleneck(p2)

        d2 = self.dec2(
            torch.cat(
                [self.up2(b), s2],
                dim=1
            )
        )

        d1 = self.dec1(
            torch.cat(
                [self.up1(d2), s1],
                dim=1
            )
        )

        residual = self.final_conv(
            self.final_up(d1)
        )

        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        return base + residual


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}"
        )

    model = RestorationNet().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# ============================================================
# LOAD INPUT
# ============================================================

def load_input(path):

    arr = np.load(path)

    arr = np.asarray(
        arr,
        dtype=np.float32
    )

    # Accept H x W or H x W x 1
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(
            f"{path.name}: expected HxW or HxWx1, "
            f"got {arr.shape}"
        )

    if not np.isfinite(arr).all():
        raise ValueError(
            f"{path.name}: input contains NaN or Inf"
        )

    tensor = torch.from_numpy(arr)

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(
        device,
        non_blocking=True
    )


# ============================================================
# INFERENCE
# ============================================================

def restore(model, inp):

    with torch.inference_mode():

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                output = model(inp)

        else:
            output = model(inp)

    output = output.float()

    # Remove batch/channel dimensions
    output = output.squeeze(0).squeeze(0)

    output = output.clamp(
        0.0,
        1.0
    )

    output = output.cpu().numpy()

    # Final safety check
    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    output = np.clip(
        output,
        0.0,
        1.0
    ).astype(np.float32)

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python run.py <input-dir> <output-dir>"
        )

        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_dir():

        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    # Create output directory automatically
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("Loading model...")

    model = load_model()

    print("Model loaded successfully.")

    files = sorted(
        input_dir.glob("*.npy")
    )

    if not files:
        raise RuntimeError(
            "No .npy files found in input directory."
        )

    print(
        f"Found {len(files)} input files."
    )

    start = time.time()

    for i, input_path in enumerate(files):

        inp = load_input(input_path)

        output = restore(
            model,
            inp
        )

        # Save with EXACT same filename
        output_path = (
            output_dir /
            input_path.name
        )

        np.save(
            output_path,
            output
        )

        if (
            (i + 1) % 50 == 0
            or i == 0
            or i + 1 == len(files)
        ):
            print(
                f"[{i + 1}/{len(files)}] "
                f"{input_path.name} "
                f"{input_path.shape if False else ''}"
                f" -> {output.shape}"
            )

    elapsed = time.time() - start

    print("\n========================================")
    print("RESTORATION COMPLETE")
    print("========================================")
    print("Images:", len(files))
    print(
        f"Total time: {elapsed:.2f} seconds"
    )
    print(
        f"Average: {elapsed / len(files):.4f} sec/image"
    )
    print(
        "Output directory:",
        output_dir
    )


if __name__ == "__main__":
    main()
