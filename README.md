 # EEG Motor Imagery — Latent Space Analysis

This project investigates unsupervised representation learning on EEG motor imagery data using a Convolutional Variational Autoencoder (CVAE), with downstream MLP classification. A supervised sequence modeling comparison is in progress.

## Dataset

This project uses the BCI Competition IV Dataset 2a, a publicly available EEG dataset recorded from 9 subjects performing 4 classes of motor imagery tasks (left hand, right hand, feet, tongue). Full dataset description is available [here](http://www.bbci.de/competition/iv/desc_2a.pdf).

Data files should be placed in the `data/` folder before running the notebooks.

## Repository Structure

```
├── data/                          # EEG data files (not tracked by git)
├── src/
│   ├── preprocessing/
│   │   └── data_prep.py           # Data preprocessing and augmentation
│   └── models/
│       └── cvae.py                # CVAE architecture definition
├── 01_cvae_experiments.ipynb  # Latent dimension and KL weight experiments
├── requirements.txt
└── README.md
```

## Notebooks

**`01_cvae_experiments.ipynb`**
Systematic experiments varying latent dimension (2–24) and KL divergence weight (1e-4 to 1e-1). Covers latent space visualization (PCA, t-SNE, UMAP), reconstruction quality assessment, and downstream MLP classification. See the notebook summary for key findings.

**`02_supervised_classification.ipynb`** *(in progress)*
CNN, CNN-LSTM, and CNN-Transformer architectures trained directly on EEG motor imagery data for comparison against the unsupervised CVAE approach.

## Getting Started

1. Clone the repository
```bash
git clone https://github.com/NayeliGuzman/eeg-motor-imagery-dl.git
cd eeg-motor-imagery-dl
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the BCI Competition IV Dataset 2a from the [official site](http://www.bbci.de/competition/iv/) and place the files in the `data/` folder.

4. Open and run `01_cvae_experiments.ipynb`

## Requirements

See `requirements.txt`. Key dependencies: TensorFlow/Keras, NumPy, Pandas, Scikit-learn, Matplotlib, UMAP-learn.
