# sp500-close-price-regressor

A PyTorch neural network that predicts the **closing price** of S&P 500 stocks from the same day's **Open, High, Low and Volume**. Hyperparameters are tuned with **Optuna**.

## Overview

| | |
|---|---|
| **Task** | Tabular regression (predict `Close`) |
| **Dataset** | [S&P 500 US Stock Market Data](https://www.kaggle.com/datasets/parsalatifi/s-and-p-500-us-stock-market-data) (Kaggle, `master_stock_data.csv`, ~2.2M rows) |
| **Inputs** | `Open`, `High`, `Low`, `Volume` |
| **Target** | `Close` |
| **Model** | MLP: Linear → BatchNorm → ReLU → Dropout blocks |
| **Libraries** | PyTorch, scikit-learn, Optuna |

## Project structure

```
.
├── 01_model_training.ipynb   # Data prep, model, training, evaluation
├── 02_optuna_tuning.ipynb    # Hyperparameter search with Optuna
├── requirements.txt
├── .gitignore
└── README.md
```

## How it works

1. **Load** `master_stock_data.csv` (downloaded with `kagglehub`).
2. **Clean**: drop `Ticker`, `Dividends`, `Stock Splits`, `Adj Close`, `Date` and rows with missing values.
3. **Split** 80% train / 20% test.
4. **Scale** features and target with `StandardScaler` (fit on train only).
5. **Train** with MSE loss; uses the GPU if available.
6. **Evaluate** with MSE, RMSE, MAE and R² on the test set (scaled units).
7. **Tune** with Optuna, maximizing R².

## Optuna search space

| Parameter | Range |
|---|---|
| Hidden layers | 1 – 5 |
| Neurons per layer | 8 – 128 (step 8) |
| Epochs | 10 – 40 (step 5) |
| Learning rate | 1e-5 – 1e-2 (log) |
| Dropout | 0.1 – 0.5 (step 0.1) |
| Batch size | 32, 64, 128 |
| Optimizer | Adam, SGD, RMSprop |
| Weight decay | 1e-5 – 1e-3 (log) |

## Results

**Best Optuna parameters:**

```python
{
    "num_hidden_layers": 2,
    "neurons_per_layer": 24,
    "epochs": 15,
    "learning_rate": 1.2392889074215176e-05,
    "dropout_rate": 0.2,
    "batch_size": 128,
    "optimizer": "SGD",
    "weight_decay": 1.145060732085222e-05,
}
```

**Optuna best trial:** R² = **0.9967** (Optuna only tracks R²).

**Final training run** (`01_model_training.ipynb`, scaled units, test set):

| MSE | RMSE | MAE | R² |
|---|---|---|---|
|0.0257 | 0.1604 | 0.0950 |** 0.9734**|
## Getting started

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

For GPU support, install PyTorch first using the command from [pytorch.org](https://pytorch.org/get-started/locally/).

1. Run the `kagglehub` download cell at the top of `01_model_training.ipynb`.
2. Set the `pd.read_csv(...)` path in both notebooks to the downloaded file.
3. Run `01_model_training.ipynb` to train and evaluate.
4. Run `02_optuna_tuning.ipynb` to repeat the hyperparameter search.

## Limitations

- **Not a forecast.** Close is predicted from the same day's Open/High/Low, which are almost perfectly correlated with it. A high R² does not mean future prices can be predicted.
- **Random split on time-series data.** Rows from the same stock and period appear in both train and test. A time-based split would be more realistic.
- **Tuning on the test set.** Optuna scores trials on the test set, so the best R² is optimistic. A separate validation set would fix this.
- **Unseeded split.** `train_test_split` has no `random_state`, so scores change between runs.

## Future work

- Add lagged features (previous closes, returns, moving averages).
- Try sequence models (LSTM / Transformer) for true next-day forecasting.
- Report errors in original price units using `y_scaler.inverse_transform`.

## License

MIT
