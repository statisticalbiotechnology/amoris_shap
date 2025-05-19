import os
import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def plot_shap_residual_vs_prediction(shap_pred_values, shap_resid_values, feature_names, outpath):
    import matplotlib.pyplot as plt

    shap_pred_values = np.array(shap_pred_values)
    shap_resid_values = np.array(shap_resid_values.values)  # shap.Explanation object

    for i, fname in enumerate(feature_names):
        plt.figure(figsize=(5, 5))
        plt.scatter(shap_pred_values[:, i], shap_resid_values[:, i], alpha=0.4)
        plt.xlabel(f"SHAP for Prediction ({fname})")
        plt.ylabel(f"SHAP for Residual ({fname})")
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.title(f"{fname}: Residual SHAP vs. Prediction SHAP")
        plt.grid(True)
        plt.savefig(f"{outpath}/shap_resid_vs_pred_{fname}.png", bbox_inches="tight")
        plt.close()

def explain_residuals(x_df, y_true, y_pred, feature_names, outdir="results", x_explain=None):
    import shap
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # Compute residuals
    residuals = y_pred - y_true

    # Train/test split for training the residual model
    x_train, _, y_train, _ = train_test_split(
        x_df.values, residuals, test_size=0.2, random_state=42
    )

    # Fit residual model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # If no x_explain provided, use part of x_df for SHAP explanation
    if x_explain is None:
        x_explain = x_df.sample(n=500, random_state=42).values  # fallback if not provided

    # SHAP explanation for residual model
    explainer = shap.Explainer(model, x_explain)
    shap_values = explainer(x_explain, check_additivity=False)

    # Plot summary of residual SHAP values
    plt.figure()
    shap.summary_plot(shap_values, features=x_explain, feature_names=feature_names, show=False)
    plt.title("SHAP Summary for Residuals (Prediction Error)")
    plt.savefig(f"{outdir}/shap_residual_summary.png", bbox_inches="tight")
    plt.close()

    print("✅ SHAP residual explanation complete.")

    return shap_values, x_explain


class WrappedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.base(x)

def plot_prediction_spread(y_true, y_pred, outpath="results/prediction_spread.png"):

    # Round age to nearest integer to group
    df = pd.DataFrame({'true_age': y_true, 'pred_age': y_pred})
    df['age_bin'] = df['true_age'].round().astype(int)

    # Compute percentiles per true age
    grouped = df.groupby('age_bin')['pred_age'].agg(
        p01=lambda x: np.percentile(x, 1),
        p10=lambda x: np.percentile(x, 10),
        p25=lambda x: np.percentile(x, 25),
        p50="median",
        p75=lambda x: np.percentile(x, 75),
        p90=lambda x: np.percentile(x, 90),
        p99=lambda x: np.percentile(x, 99),
        count="count"
    ).reset_index()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(grouped['age_bin'], grouped['p50'], label="Median", color="black")
    plt.fill_between(grouped['age_bin'], grouped['p25'], grouped['p75'], alpha=0.3, label="25–75%")
    plt.fill_between(grouped['age_bin'], grouped['p10'], grouped['p90'], alpha=0.2, label="10–90%")
    plt.fill_between(grouped['age_bin'], grouped['p01'], grouped['p99'], alpha=0.1, label="1–99%")

    plt.xlabel("True Age (binned)")
    plt.ylabel("Predicted Age")
    plt.title("Distribution of Predicted Ages by True Age")
    plt.legend()
    plt.grid(True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def plot_prediction_diagnostics(model, x_tensor, y_true, feature_names, outdir):
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor).squeeze().numpy()

    # Predicted vs True
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    sns.regplot(x=y_true, y=y_pred, scatter=False, color="red")
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title("Predicted vs True Age")
    plt.savefig(f"{outdir}/predicted_vs_true.png", bbox_inches="tight")
    plt.close()

    # Spread plot
    plot_prediction_spread(y_true, y_pred, outpath=f"{outdir}/prediction_spread.png")
    return y_pred

def explain_prediction_shap(model, x_tensor, feature_names, outdir):
    x_explain = x_tensor[:500]
    x_background = x_tensor[:100]
    wrapped_model = WrappedModel(model)

    x_explain_np = x_explain.numpy()
    x_background_np = x_background.numpy()

    def predict_fn(x_np):
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            return wrapped_model(x_tensor).detach().numpy()

    explainer = shap.KernelExplainer(predict_fn, x_background_np)
    shap_values = explainer.shap_values(x_explain_np, nsamples="auto")

    # Clean up shape
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)
    if shap_values.shape != x_explain_np.shape:
        raise ValueError(f"SHAP shape mismatch: expected {x_explain_np.shape}, got {shap_values.shape}")

    # Summary plot
    shap.summary_plot(shap_values, features=x_explain_np, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig(f"{outdir}/shap_summary.png", bbox_inches="tight")
    plt.close()

    # Per-feature plots
    for i, name in enumerate(feature_names):
        shap.dependence_plot(i, shap_values, x_explain_np, feature_names=feature_names, show=False)
        plt.title(f"SHAP Feature: {name}")
        plt.savefig(f"{outdir}/shap_feature_{name}.png", bbox_inches="tight")
        plt.close()

    return shap_values, x_explain_np

def generate_html_report(feature_names, outdir):
    with open(f"{outdir}/report.html", "w") as f:
        f.write("<html><head><title>SHAP & Regression Report</title></head><body>\n")
        f.write("<h1>Predicted vs True Age</h1>\n")
        f.write('<img src="predicted_vs_true.png" width="600"><br><hr>\n')
        f.write('<h1>Prediction Spread by True Age</h1>\n')
        f.write('<img src="prediction_spread.png" width="600"><br><hr>\n')
        f.write("<h1>SHAP Summary Plot</h1>\n")
        f.write('<img src="shap_summary.png" width="600"><br><hr>\n')
        f.write("<h1>Feature SHAP Effects</h1>\n")
        for name in feature_names:
            f.write(f"<h2>{name}</h2>\n")
            f.write(f'<img src="shap_feature_{name}.png" width="600"><br>\n')
        f.write("<h1>SHAP Residual Explanation</h1>\n")
        f.write('<img src="shap_residual_summary.png" width="600"><br><hr>\n')
        f.write('<h1>Residual vs Prediction SHAP Correlation</h1>\n')
        f.write('<p>SHAP values for residual error vs. prediction per feature.</p>\n')
        for name in feature_names:
            f.write(f"<h2>{name}</h2>\n")
            f.write(f'<img src="shap_resid_vs_pred_{name}.png" width="600"><br>\n')
        f.write("</body></html>")
    print(f"✅ SHAP report complete. View it in: {outdir}/report.html")


def explain_model_with_shap(model, x_tensor, y_true, feature_names, outdir="results", model_name="amoris"):
    # 1. Run prediction and basic plots
    y_pred = plot_prediction_diagnostics(model, x_tensor, y_true, feature_names, outdir)

    # 2. Explain prediction SHAP
    shap_pred_values, x_explain_np = explain_prediction_shap(model, x_tensor, feature_names, outdir)

    # 3. Explain residual SHAP
    shap_resid_values, _ = explain_residuals(
        x_df=pd.DataFrame(x_tensor.numpy(), columns=feature_names),
        y_true=y_true,
        y_pred=y_pred,
        feature_names=feature_names,
        outdir=outdir,
        x_explain=x_explain_np
    )

    # 4. Plot residual vs prediction SHAPs
    plot_shap_residual_vs_prediction(
        shap_pred_values=shap_pred_values,
        shap_resid_values=shap_resid_values,
        feature_names=feature_names,
        outpath=outdir
    )

    # 5. Generate report
    generate_html_report(feature_names, outdir)
