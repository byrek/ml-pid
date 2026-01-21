import pandas as pd
import mlflow
import mlflow.pytorch
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- CONFIG ---
lags_values = list(range(3, 12,1))  #od 1 do 8 co 1

mlflow.set_experiment("PID ML - Linear")

# direcories
models_dir = './models'
os.makedirs(models_dir, exist_ok=True)

# device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Linear model with PyTorch ---
class LinearRegressorTorch(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

# reading data
df = pd.read_csv('./data/motor_telemetry.csv')
print("\nFirst 5 rows of data:")
print(df.head())
print("Amount of records:", len(df))

# --- lag loop ---
for lags in lags_values:
    run_name = f"PyTorchLinear{lags}"
    
    with mlflow.start_run(run_name=run_name):
        
        # logging params
        mlflow.log_param("lags", lags)
        mlflow.log_param("model_type", "PyTorch Linear")
        mlflow.log_param("sampling_ms", 1)
        mlflow.log_param("normalization", "StandardScaler")
        mlflow.log_param("device", str(device))

        # creating lags
        df_lag = df.copy()
        for col in ['ERROR', 'CURRENT']:
            for lag in range(1, lags + 1):
                df_lag[f'{col}_lag{lag}'] = df_lag[col].shift(lag)
        
        # remove NaN
        df_ready = df_lag.dropna().reset_index(drop=True)
        mlflow.log_param("samples_after_dropna", len(df_ready))

        # features i target
        feature_columns = ['ERROR', 'CURRENT'] + \
                          [col for col in df_ready.columns if '_lag' in col]
        
        X = df_ready[feature_columns].values.astype(np.float32)
        y = df_ready['VREF'].values.astype(np.float32).reshape(-1, 1)

        n_features = X.shape[1]
        mlflow.log_param("features_count", n_features)

        # data split
        n = len(X)
        train_end = int(n * 0.80)
        val_end = train_end + int(n * 0.15)

        X_train_raw = X[:train_end]
        X_val_raw = X[train_end:val_end]
        X_test_raw = X[val_end:]

        y_train_raw = y[:train_end]
        y_val_raw = y[train_end:val_end]
        y_test_raw = y[val_end:]

        # scalers
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        scaler_X.fit(X_train_raw)
        scaler_y.fit(y_train_raw)

        X_train = scaler_X.transform(X_train_raw)
        X_val = scaler_X.transform(X_val_raw)
        X_test = scaler_X.transform(X_test_raw)

        y_train = scaler_y.transform(y_train_raw).ravel()
        y_val = scaler_y.transform(y_val_raw).ravel()

        # dataLoader
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # model & optimizer
        model = LinearRegressorTorch(n_features=n_features).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # training – 50 epoch, progress 10
        epochs = 50
        print_every = 10

        print(f"\nTraining the model for {lags} lags in {epochs} epochs...")
        model.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X).ravel()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)

            if epoch % print_every == 0 or epoch == epochs:
                print(f"  Epoch {epoch:2d}/{epochs} | Avg Loss: {epoch_loss:.6f}")

        print(f"Training done for {lags} lags.\n")

        # prediction
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val).to(device)
            X_test_tensor = torch.from_numpy(X_test).to(device)

            y_val_pred_scaled = model(X_val_tensor).cpu().numpy().ravel()
            y_test_pred_scaled = model(X_test_tensor).cpu().numpy().ravel()

        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        y_test_pred_rounded = np.round(y_test_pred)

        # metrics
        val_mse = mean_squared_error(y_val_raw.ravel(), y_val_pred)
        val_r2 = r2_score(y_val_raw.ravel(), y_val_pred)
        test_mse = mean_squared_error(y_test_raw.ravel(), y_test_pred)
        test_r2 = r2_score(y_test_raw.ravel(), y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mse_rounded = mean_squared_error(y_test_raw.ravel(), y_test_pred_rounded)
        test_rmse_rounded = np.sqrt(test_mse_rounded)

        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mse_rounded", test_mse_rounded)
        mlflow.log_metric("test_rmse_rounded", test_rmse_rounded)

        # logging model
        mlflow.pytorch.log_model(
            pytorch_model=model.cpu(),
            name="model", 
            registered_model_name=run_name
        )

        # export to ONNX
        onnx_path = f"{models_dir}/{run_name}.onnx"
        dummy_input = torch.randn(1, n_features)

        torch.onnx.export(
            model.cpu(),
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        mlflow.log_artifact(onnx_path)

        # recording scalers
        artifacts_path = f"{models_dir}/{run_name}_artifacts.pkl"
        joblib.dump({
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_columns': feature_columns,
            'lags': lags
        }, artifacts_path)
        mlflow.log_artifact(artifacts_path)

        print(f"lags={lags} | Val R²={val_r2:.4f} | Test R²={test_r2:.4f} | Test RMSE_rounded={test_rmse_rounded:.4f}")
        print(f"   → Model recorded in MLflow")
        print(f"   → ONNX ready: {onnx_path}\n")

print("All models were trained and are saved!")