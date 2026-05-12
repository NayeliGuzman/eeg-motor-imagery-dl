 # EEG Motor Imagery:  Latent Space Analysis

This project investigates unsupervised representation learning on EEG motor imagery data using a Convolutional Variational Autoencoder (CVAE), with downstream MLP classification. A supervised sequence modeling comparison is in progress.

## Dataset

This project uses the BCI Competition IV Dataset 2a, a publicly available EEG dataset recorded from 9 subjects performing 4 classes of motor imagery tasks (left hand, right hand, feet, tongue). Full dataset description is available [here](http://www.bbci.de/competition/iv/desc_2a.pdf).

Data files should be placed in the `data/` folder before running the notebooks.

## Notebooks

**`01_cvae_experiments.ipynb`**
Systematic experiments varying latent dimension (2–24) and KL divergence weight (1e-4 to 1e-1). Covers latent space visualization (PCA, t-SNE, UMAP), reconstruction quality assessment, and downstream MLP classification. See the notebook summary for key findings.


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
