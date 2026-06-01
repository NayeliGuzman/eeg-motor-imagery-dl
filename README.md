 # EEG Motor Imagery:  Classification Task and Latent Space Analysis

This project investigates deep learning approaches for EEG motor imagery classification using the 
BCI Competition IV Dataset 2a. Experiments include unsupervised representation learning via a 
Convolutional Variational Autoencoder (CVAE) with downstream MLP classification, and supervised 
classification using CNN-LSTM, CNN-LSTM-Residual, CNN-Conformer (Keras), and ShallowConvNet 
(PyTorch) architectures. A raw data preprocessing pipeline from `.gdf` to `.npy` format is also 
demonstrated.

## I. Dataset

This project uses the BCI Competition IV Dataset 2a, a publicly available EEG dataset recorded from 9 subjects performing 4 classes of motor imagery tasks (left hand, right hand, feet, tongue). Full dataset description is available [here](http://www.bbci.de/competition/iv/desc_2a.pdf).


## II. Preprocessing

### Raw Data Pipeline
Download the BCI Competition IV Dataset 2a from the [official site](http://www.bbci.de/competition/iv/) and place the `.gdf` files in the `data/BCICIV_2a_gdf/` folder.

`src/preprocessing/process_raw_data.py` demonstrates a full preprocessing pipeline from raw `.gdf` files to `.npy` format. To run it:

```bash
src/preprocessing/run_preprocessing.sh data/BCICIV_2a_gdf data/processed
```
This will save the processed `.npy` files to `data/processed/`. Note: Subject 4 is excluded 
due to missing motor imagery classes caused by a technical recording problem documented in the 
dataset description (Brunner et al., 2008).

### Notebook Preprocessing
All notebooks use previously preprocessed `.npy` files as their starting point. Those files are not included here due to size contraits.  Further preprocessing 
varies by architecture and notebook:

- **`00_baseline_cnn_lstm.ipynb`, `01_cvae_experiments.ipynb`, `02_cnn_experiments.ipynb`** — 
  additional preprocessing is handled by `src/preprocessing/data_preprocessing.py`
- **`03_shallowconvnet_pytorch.ipynb`** — additional preprocessing is handled entirely within the notebook 

## III. Notebooks
**`00_baseline_cnn_lstm.ipynb`**
This notebook demonstrates the use of data_preprocessing.py to process EEG data and cnn_lstm.py for building a CNN-LSTM classifier.

**`01_cvae_experiments.ipynb`**
Systematic experiments varying latent dimension (2–24) and KL divergence weight (1e-4 to 1e-1). Covers latent space visualization (PCA, t-SNE, UMAP), reconstruction quality assessment, and downstream MLP classification. See the notebook summary for key findings.

**`02_cnn_experiments.ipynb`** Three models: CNN-LSTM, CNN-LSTM-RESIDUAL, and CNN-CONFORMER architectures are trained to perform classification. The assessment is based on accuracy and loss. The model architectures can be found in `src/models`. 

**`03_shallowconvnet_pytorch.ipynb`** This notebook uses PyTorch to demonstrate the shallow conv net architecture as a classifier for this dataset.  

## VI. Getting Started

1. Clone the repository
```bash
git clone https://github.com/NayeliGuzman/eeg-motor-imagery-dl.git
cd eeg-motor-imagery-dl
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the BCI Competition IV Dataset 2a from the [official site](http://www.bbci.de/competition/iv/) 
and place the `.gdf` files in `data/BCICIV_2a_gdf/`. See the [Preprocessing](#ii-preprocessing) 
section for details on generating the `.npy` files.

4. Open and run any of the notebooks 

## Requirements

See `requirements.txt`. Key dependencies: TensorFlow/Keras, PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib, MNE, UMAP-learn.

## References

1. Clemens Brunner, Robert Leeb, Gernot Müller-Putz 2024 BCI Competition 2008–Graz data set A. Available at: https://dx.doi.org/10.21227/katb-zv89


2. Schirrmeister RT, Springenberg JT, Fiederer LDJ, Glasstetter M, Eggensperger K, Tangermann M, Hutter F, Burgard W, Ball T. Deep learning with convolutional neural networks for EEG decoding and visualization. Hum Brain Mapp. 2017 Nov;38(11):5391-5420. doi: 10.1002/hbm.23730. Epub 2017 Aug 7. PMID: 28782865; PMCID: PMC5655781.