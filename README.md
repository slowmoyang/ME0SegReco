# ME0SegReco

ME0 Segment Reconstruction using Deep Learning

## Overview

This repository contains implementations of deep learning models for ME0 segment reconstruction in high energy physics. It includes CNN-based and Transformer-based approaches for segment reconstruction from ME0 detector data.

## Models

- **ME0SegCNN3d**: 3D Convolutional Neural Network for image-based segment reconstruction
- **ME0Transformer**: Transformer-based model for sequence-based segment reconstruction

## Installation

### Prerequisites

- Python 3.11
- micromamba or conda

### Setup Environment

1. Create the conda environment from the provided configuration:

```bash
micromamba env create -f environment.yaml
```

2. Activate the environment:

```bash
micromamba activate me0segreco-py311
```

3. Set up the Python path:

```bash
source setup.sh
```

## Usage

### Training

Train a model using the provided configuration files:

```bash
python train.py --config config/cnn.yaml
```

or for the Transformer model:

```bash
python train.py --config config/transformer.yaml
```

### Quick Sanity Check

Run a quick sanity check with reduced dataset and epochs:

```bash
./sanity-check.sh cnn
```

## Configuration

Configuration files are located in the `config/` directory:

- `cnn.yaml`: Configuration for CNN-based model
- `transformer.yaml`: Configuration for Transformer-based model

Note: Update the file paths in configuration files to match your data location.

## Examples

Jupyter notebooks demonstrating the models and metrics are available in the `examples/` directory:

- `00.dataset.ipynb`: Dataset exploration
- `01.ME0SegCNN3d.ipynb`: CNN model demonstration
- `02.ME0Transformer.ipynb`: Transformer model demonstration
- `03.Metrics.ipynb`: Metrics evaluation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Yeonju Kim
