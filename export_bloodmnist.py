#!/usr/bin/env python3
"""
================================================================
 export_bloodmnist.py  —  Exporta BloodMNIST al formato binario
 que lee el loader C++ de este proyecto.
================================================================

Formato del archivo .bin generado
──────────────────────────────────
 Header (4 ints de 32 bits, little-endian):
   [0]  nSamples   — número de muestras
   [1]  width      — ancho de la imagen  (28)
   [2]  height     — alto  de la imagen  (28)
   [3]  channels   — número de canales   (3, RGB)

 Por cada muestra (en orden):
   [int32]  label            — clase (0-7)
   [float32 × W×H×C]  píxeles normalizados [0,1]
              layout: channel-first (C, H, W) → aplanado

Uso:
    pip install medmnist pillow numpy
    python export_bloodmnist.py

Salida:
    data/bloodmnist_train.bin
    data/bloodmnist_val.bin
    data/bloodmnist_test.bin
    data/bloodmnist_info.txt   ← metadatos legibles
"""

import sys
import struct
import os
import numpy as np

# ── Instalación automática de medmnist si no está ────────────────────────────
try:
    import medmnist
except ImportError:
    print("[INFO] medmnist no encontrado. Instalando con pip...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "medmnist"])
    import medmnist

from medmnist import BloodMNIST, INFO

# ── Configuración ─────────────────────────────────────────────────────────────
OUTPUT_DIR  = "data"           # carpeta de salida (se crea si no existe)
IMAGE_SIZE  = 28               # BloodMNIST usa 28×28 por defecto
DOWNLOAD    = True             # descarga automática si no está en caché

# Clases de BloodMNIST (8 tipos de células sanguíneas)
CLASSES = [
    "basophil",       # 0
    "eosinophil",     # 1
    "erythroblast",   # 2
    "ig",             # 3  (immature granulocytes)
    "lymphocyte",     # 4
    "monocyte",       # 5
    "neutrophil",     # 6
    "platelet",       # 7
]
NUM_CLASSES = len(CLASSES)


def export_split(dataset, output_path: str):
    """
    Escribe un split (train/val/test) al formato binario del proyecto.

    Estructura por muestra:
        int32   label
        float32 × (C × H × W)   píxeles en [0, 1], orden (C, H, W) aplanado
    """
    images = dataset.imgs    # numpy (N, H, W, C)  uint8  [0,255]
    labels = dataset.labels  # numpy (N, 1)         int64

    N, H, W, C = images.shape

    print(f"  → {output_path}")
    print(f"     muestras={N}  H={H}  W={W}  C={C}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        # ── Header ──
        f.write(struct.pack("<i", N))
        f.write(struct.pack("<i", W))
        f.write(struct.pack("<i", H))
        f.write(struct.pack("<i", C))

        # ── Muestras ──
        for i in range(N):
            label = int(labels[i][0])
            # Convertir a float32 [0,1] y reordenar a (C, H, W)
            img = images[i].astype(np.float32) / 255.0   # (H, W, C)
            img_chw = img.transpose(2, 0, 1)              # (C, H, W)
            pixels  = img_chw.flatten()                   # C*H*W floats

            f.write(struct.pack("<i", label))
            f.write(pixels.tobytes())   # float32 little-endian

    print(f"     ✓ escrito ({os.path.getsize(output_path) / 1024:.1f} KB)")


def write_info(output_dir: str):
    """Escribe un archivo de texto con los metadatos del dataset."""
    info = INFO["bloodmnist"]
    path = os.path.join(output_dir, "bloodmnist_info.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== BloodMNIST Dataset Info ===\n\n")
        f.write(f"Descripción   : {info.get('description', 'N/A')}\n")
        f.write(f"Num. clases   : {NUM_CLASSES}\n")
        f.write(f"Tamaño imagen : 28 × 28 × 3 (RGB)\n\n")
        f.write("Clases:\n")
        for i, cls in enumerate(CLASSES):
            f.write(f"  {i}: {cls}\n")
        f.write("\nFormato binario (.bin):\n")
        f.write("  Header: [nSamples(i32), width(i32), height(i32), channels(i32)]\n")
        f.write("  Por muestra: [label(i32), píxeles(f32 × C×H×W)]\n")
        f.write("  Píxeles normalizados [0,1], layout channel-first (C,H,W)\n")
    print(f"  → {path}  ✓")


def main():
    print("=" * 60)
    print(" Exportador BloodMNIST → binario C++")
    print("=" * 60)
    print(f"medmnist versión: {medmnist.__version__}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    splits = {
        "train" : BloodMNIST(split="train", download=DOWNLOAD, size=IMAGE_SIZE),
        "val"   : BloodMNIST(split="val",   download=DOWNLOAD, size=IMAGE_SIZE),
        "test"  : BloodMNIST(split="test",  download=DOWNLOAD, size=IMAGE_SIZE),
    }

    for split_name, dataset in splits.items():
        print(f"\n[{split_name.upper()}]")
        out_path = os.path.join(OUTPUT_DIR, f"bloodmnist_{split_name}.bin")
        export_split(dataset, out_path)

    print("\n[INFO]")
    write_info(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(" Exportación completa.")
    print(f" Archivos en: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)
    print()
    print("En C++:")
    print('  Dataset train = loadBloodMNIST("data/bloodmnist_train.bin");')
    print(f'  // {NUM_CLASSES} clases, imágenes 28×28×3')


if __name__ == "__main__":
    main()
