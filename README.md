# Credit Card Fraud Detection — Colab + GitHub + Streamlit

This starter project trains a machine-learning classifier from `credit_card_fraud_custom.csv` and serves predictions in a Streamlit web app.

## Files
- `credit_card_fraud_custom.csv` — dataset supplied with this project
- `fraud_detection_colab.ipynb` — notebook to upload/open in Google Colab
- `train_model.py` — trains and saves the model as `fraud_model.joblib`
- `app.py` — Streamlit user interface
- `requirements.txt` — Python dependencies

## Option A: Train in Google Colab
1. Upload `fraud_detection_colab.ipynb` to Google Colab, or open it from your GitHub repository.
2. Run the notebook cells in order. The notebook installs dependencies, reads the CSV, trains the model, prints evaluation metrics, and downloads `fraud_model.joblib`.
3. Put the downloaded `fraud_model.joblib` in the repository root before deploying the Streamlit app.

## Option B: Train locally
```bash
pip install -r requirements.txt
python train_model.py
```

## Run Streamlit locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deploy with GitHub + Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `train_model.py`, `requirements.txt`, `credit_card_fraud_custom.csv`, and the generated `fraud_model.joblib`.
2. Open Streamlit Community Cloud and deploy the repository, selecting `app.py` as the entry point.
3. If you train the model in Colab, upload the resulting `fraud_model.joblib` to the repository. Alternatively, run training locally and upload the model file.

## Important notes
- `IsFraud` is the target label; `TransactionID` is excluded as an identifier.
- The notebook uses a random train/test split for a starter demonstration. For real-world fraud detection, evaluate temporal leakage, false-negative costs, class imbalance, and drift.
- The supplied CSV has a balanced target distribution, which may not reflect real fraud prevalence. Model performance on this dataset should not be interpreted as real-world financial performance.
- This is an educational starter project, not a production fraud system. Do not deploy it for real financial decisions without robust validation, monitoring, security, and expert review.
