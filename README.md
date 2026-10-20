# LiDAR Point Cloud Classification (TerraScan Class Schema)

Semantic classification of airborne LiDAR point clouds into TerraScan-style classes (ground, vegetation, buildings, etc.), built by fine-tuning the open-source [Myria3D](https://github.com/IGNF/myria3d) framework.

## Overview

Manual classification of LiDAR data is slow and expensive. This project automates it by fine-tuning a pretrained deep learning model to predict TerraScan-compatible classes directly on raw point clouds.

**What I did:**
- Mapped the dataset labels to the TerraScan class schema ([N] classes)
- Prepared the data: tiling, sampling, and label cleanup
- Fine-tuned Myria3D with custom configs for this class schema
- Evaluated per-class and overall performance on a held-out set
- [Add anything else you actually did: augmentation, class balancing, post-processing, etc.]

## Results

| Metric | Value |
|---|---|
| Overall accuracy | [XX.X %] |
| mIoU | [XX.X %] |
| Training / validation split | [e.g., 80 / 20 by tile] |

**Per-class IoU**

| Class | IoU |
|---|---|
| Ground | [XX.X %] |
| Low / Medium / High vegetation | [XX.X %] |
| Building | [XX.X %] |
| [Other classes] | [XX.X %] |

[Add one image here: a before/after or prediction-vs-ground-truth screenshot of a tile.]

```
![Prediction vs ground truth](docs/prediction_example.png)
```

## Tech Stack

- Python, PyTorch, PyTorch Lightning, Hydra
- [laspy / PDAL / other libraries you used]
- [GPU used for training, e.g., NVIDIA T400 4GB]

## Project Structure

```
.
├── configs/          # Hydra configs for training / fine-tuning
├── scripts/          # Data preparation and evaluation scripts
├── docs/             # Images and notes
├── requirements.txt
└── README.md
```

[Edit this tree to match your actual folders.]

## Getting Started

```bash
# 1. Clone
git clone https://github.com/mr-liki/lidar-terrascan-classification.git
cd lidar-terrascan-classification

# 2. Create environment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Fine-tune
python run.py --config-name [your_config]

# 4. Evaluate / predict
python run.py --config-name [your_predict_config]
```

[Replace the commands with the exact ones you use.]

## Data

The training data is **not included** in this repository. [One line on the data type, e.g., "Airborne LiDAR tiles in LAS/LAZ format, classified in TerraScan."] To reproduce, supply your own LAS/LAZ data labeled with the class schema in `configs/`.

## Roadmap

- [x] Fine-tune Myria3D on TerraScan class schema
- [ ] Improve weak classes ([list them])
- [ ] Reimplement the model architecture from scratch to understand and customize it
- [ ] Add an inference script for full-block prediction

## Acknowledgements

This project builds on [Myria3D](https://github.com/IGNF/myria3d) by IGN (Institut National de l'Information Géographique et Forestière), licensed under BSD-3-Clause. Fine-tuning, data preparation, and TerraScan class mapping are my own work.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).

## Author

**Likhith R** · [GitHub](https://github.com/mr-liki) · [LinkedIn]([your LinkedIn URL])
