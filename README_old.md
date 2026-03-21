# UNSW-NB15 Intrusion Detection Project

This project has been restructured to follow a simpler, flat layout:

```text
project/
│
├── data/
├── notebooks/
├── models/
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── inference.py
├── demo.py
├── requirements.txt
└── README.md
```

## What is included
- `data/`: training and testing CSV files
- `notebooks/`: the project notebooks, including the polished and hybrid versions
- `models/`: saved baseline artifacts from the current project version
- `dataset.py`: data loading and preprocessing helpers
- `model.py`: PyTorch model definitions
- `train.py`: trains Logistic Regression, Random Forest, CNN, and Autoencoder
- `evaluate.py`: prints saved metrics
- `inference.py`: runs batch inference on a CSV file
- `demo.py`: Streamlit demo entry point

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train or refresh models:

```bash
python train.py
```

View metrics:

```bash
python evaluate.py
```

Run batch inference:

```bash
python inference.py --input data/UNSW_NB15_testing-set.csv --output predictions.csv --model random_forest
```

Launch the demo:

```bash
streamlit run demo.py
```

## Notes
- `id` and `attack_cat` are dropped during preprocessing.
- The flat layout is intended to be easier to present and submit.
- Existing notebook files are preserved in `notebooks/`.


## Hybrid Fusion Framework

This project now includes the proposed hybrid framework:

- **Supervised classification:** Random Forest
- **Unsupervised anomaly detection:** Dense Autoencoder
- **Fusion strategy:**  
  `Final Score = α × Supervised Probability + (1 − α) × Anomaly Score`

Where:
- `α` is a weighting factor tuned on a validation split
- the final threshold is also tuned on validation data

This allows the system to detect:
- **known attacks** through the Random Forest branch
- **unknown anomalies** through the Dense Autoencoder branch

### Hybrid outputs
The hybrid pipeline produces:
- `supervised_score`
- `anomaly_score`
- `fusion_score`
- `prediction`



## Independent notebook

A new notebook was added:

- `notebooks/UNSW_NB15_Independent_Hybrid_Framework.ipynb`

This notebook is self-contained and follows the same logic as:

- `dataset.py`
- `model.py`
- `train.py`
- `inference.py`

It also displays the full results directly inside the notebook.


## Improved Dense Autoencoder

The project now uses an improved Dense Autoencoder with:

- deeper encoder and decoder layers
- batch normalization
- dropout regularization
- longer training
- validation-based threshold tuning
- log-scaled normalized reconstruction error

These updates improve anomaly-score quality and make the hybrid fusion branch stronger.


## Fixed normalization and recall-constrained thresholding

The Dense Autoencoder and Hybrid Fusion pipeline now use:

- fixed anomaly-score normalization based on validation-set score min and max
- recall-constrained threshold tuning
- shared anomaly-score scaling across validation, test, inference, and demo flows

This keeps thresholds consistent between validation and test and improves practical anomaly detection behavior.
