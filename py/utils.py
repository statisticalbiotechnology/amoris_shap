import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    columns_of_interest = ["sampleID", "status", "albumin", "alp", "lymph", "mcv",
                           "lncreat", "lncrp", "hba1c", "wbc", "rdw", "age"]
    df = df[columns_of_interest].dropna()
    ids = df['sampleID']
    status = df['status']
    ages = df['age'].values
    df_normalized = df.drop(['sampleID', 'status'], axis=1)
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(df_normalized)
    df_normalized = pd.DataFrame(x_normalized, columns=df_normalized.columns)
    return df_normalized, scaler, ids, status, ages
