import os
import logging
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import random

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger()

class QuantizedLinearRegression(nn.Module):
    def __init__(self, input_dim, weight, bias):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.linear.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32).unsqueeze(0))
        self.linear.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)

def quantize_linear8(params):
    params = params.astype(np.float32)
    qmin, qmax = 0, 255
    scale = (params.max() - params.min()) / (qmax - qmin) if params.max() != params.min() else 1.0
    zero_point = np.round(-params.min() / scale).astype(np.int32)
    quantized = np.clip(np.round(params / scale + zero_point), qmin, qmax).astype(np.uint8)
    dequantized = scale * (quantized.astype(np.float32) - zero_point)
    error = np.mean(np.abs(params - dequantized))
    logger.info(f"8-bit quantization error (MAE): {error:.8e}")
    return quantized, scale, zero_point, dequantized

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        preds = model(inputs)
        r2 = r2_score(targets.numpy(), preds.numpy())
        # RMSE using square root of MSE, works with all sklearn versions!
        rmse = np.sqrt(mean_squared_error(targets.numpy(), preds.numpy()))
    return r2, rmse

def main():
    logger.info("Starting quantization...")

    model_path = "models/linear_model.joblib"
    test_data_path = "models/test_data.joblib"
    quantized_model_path = "models/quantized_pytorch_model.pth"

    if not os.path.exists(model_path) or not os.path.exists(test_data_path):
        logger.error("Required model or test data missing. Run train.py first.")
        return

    # Load sklearn model and test data
    model_sklearn = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    weights = model_sklearn.coef_
    bias = model_sklearn.intercept_
    logger.info(f"Loaded model coef shape: {weights.shape}, bias: {bias:.6f}")

    # Quantize weights and bias
    q_w, scale_w, zp_w, dq_w = quantize_linear8(weights)
    q_b, scale_b, zp_b, dq_b_arr = quantize_linear8(np.array([bias]))
    dq_b = dq_b_arr[0]

    # Build PyTorch model with dequantized parameters
    quant_model = QuantizedLinearRegression(len(weights), dq_w, dq_b)
    os.makedirs("models", exist_ok=True)
    torch.save(quant_model.state_dict(), quantized_model_path)
    logger.info(f"Quantized PyTorch model saved to {quantized_model_path}")

    # Evaluate original and quantized models
    y_pred_orig = model_sklearn.predict(X_test)
    r2_orig = r2_score(y_test, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_orig))
    r2_quant, rmse_quant = evaluate_model(quant_model, X_test, y_test)

    # File size comparison
    size_orig_kb = os.path.getsize(model_path) / 1024
    size_quant_kb = os.path.getsize(quantized_model_path) / 1024
    compression_ratio = size_orig_kb / size_quant_kb if size_quant_kb else float('inf')

    # Output results
    print("\nFINAL COMPARISON TABLE")
    print(f"{'Metric':<20}{'Original':>15}{'Quantized':>15}")
    print("-" * 50)
    print(f"{'R² Score':<20}{r2_orig:15.6f}{r2_quant:15.6f}")
    print(f"{'RMSE':<20}{rmse_orig:15.6f}{rmse_quant:15.6f}")
    print(f"{'File Size (KB)':<20}{size_orig_kb:15.3f}{size_quant_kb:15.3f}")
    print(f"{'Compression Ratio':<20}{compression_ratio:15.2f}x")

    # Save comparison for reporting
    comparison = {
        "r2_original": r2_orig,
        "rmse_original": rmse_orig,
        "r2_quantized": r2_quant,
        "rmse_quantized": rmse_quant,
        "size_original_kb": size_orig_kb,
        "size_quantized_kb": size_quant_kb,
        "compression_ratio": compression_ratio,
    }
    joblib.dump(comparison, "models/comparison_results.joblib")

    logger.info("Quantization completed successfully.")

if __name__ == "__main__":
    main()