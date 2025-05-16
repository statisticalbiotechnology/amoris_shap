from model import AgeRegressor
from utils import load_data
from shap_utils import explain_model_with_shap
import torch

def main():
    df_norm, _, _, _, ages = load_data("../data/NH3.csv")
    x = torch.tensor(df_norm.drop("age", axis=1).values, dtype=torch.float32)
    feature_names = df_norm.drop("age", axis=1).columns

    # Load trained model
    model = AgeRegressor(input_dim=x.shape[1])
    checkpoint = torch.load("amoris_model.ckpt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    explain_model_with_shap(model.model, x, ages, feature_names, outdir="../results")

if __name__ == "__main__":
    main()
