# Import necessary libraries
import os
import re
from typing import List, Optional
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
import zipfile
import shutil
import typer


# Define the function to generate images and save them as a ZIP folder
def generate_images_zip(
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
):
    # Load the network pickle
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    # Initialize a ZIP file for saving images
    os.makedirs(
        outdir, exist_ok=True
    )  # Create the output directory if it doesn't exist
    zip_filename = os.path.join(outdir, "generated_images.zip")
    zip_file = zipfile.ZipFile(zip_filename, "w")

    # Labels
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            print(
                "Must specify class label with --class when using a conditional network"
            )
            return
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print(
                "Warning: --class=lbl ignored when running on an unconditional network"
            )

    # Generate images
    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_filename = os.path.join(outdir, f"seed{seed:04d}.png")
        PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(image_filename)
        zip_file.write(image_filename, os.path.basename(image_filename))

    # Close the ZIP file
    zip_file.close()

    print(f"Generated images saved as ZIP folder: {zip_filename}")


app = typer.Typer()


@app.command()
def generate(
    network_pkl: str = typer.Option(
        "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl",
        help="Network pickle filename",
    ),
    seeds: List[int] = typer.Option(None, help="List of random seeds"),
    truncation_psi: float = typer.Option(1, help="Truncation psi", show_default=True),
    noise_mode: str = typer.Option("const", help="Noise mode", show_default=True),
    outdir: str = typer.Option(
        "/data/1birinci/datasets/generated_images",
        help="Where to save the output ZIP file",
        show_default=True,
    ),
    class_idx: Optional[int] = typer.Option(
        None, help="Class label (unconditional if not specified)"
    ),
    projected_w: Optional[str] = typer.Option(
        None, help="Projection result file", metavar="FILE"
    ),
):
    generate_images_zip(
        network_pkl, seeds, truncation_psi, noise_mode, outdir, class_idx, projected_w
    )


if __name__ == "__main__":
    app()
