import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

RAW_PATH = r"C:\Users\sojay\DS\VS_Code\Project_4\data\swiggy.csv"


# =========================================================
# CLEAN DATA
# =========================================================
def clean_data():

    df = pd.read_csv(RAW_PATH)

    df = df.drop_duplicates()
    df = df[df["cuisine"]!='8:15 To 11:30 Pm']

    # -------------------------
    # rating
    # -------------------------
    df["rating"] = df["rating"].replace("--", np.nan)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = df["rating"].fillna(df["rating"].mean())

    # -------------------------
    # rating_count
    # -------------------------
    def extract_count(x):
        if isinstance(x, str):
            nums = re.findall(r"\d+", x)
            return int(nums[0]) if nums else 0
        return 0

    df["rating_count"] = df["rating_count"].apply(extract_count)
    df["rating_count"] = df["rating_count"].astype(int)

    # -------------------------
    # cost
    # -------------------------
    df["cost"] = (
        df["cost"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
    )

    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["cost"] = df["cost"].fillna(df["cost"].median()).astype(int)

    df.reset_index(drop=True, inplace=True)

    df.to_csv(
        r"C:\Users\sojay\DS\VS_Code\Project_4\data\cleaned_data.csv",
        index=False
    )

    print("Cleaned data saved")

    return df


# =========================================================
# ENCODING  (🔥 FIXED VERSION)
# =========================================================
def encode_data(df):

    df = df.copy()  # safety

    numerical_cols = ["rating", "rating_count", "cost"]

    # ---------------------------------
    # numeric (3 columns)
    # ---------------------------------
    numeric = df[numerical_cols].to_numpy(dtype=np.float32)

    # =================================
    # CITY → LabelEncoding (1 column)
    # =================================
    city_encoder = LabelEncoder()
    city_encoded = city_encoder.fit_transform(df["city"]).astype(np.float32)
    city_encoded = city_encoded.reshape(-1, 1)

    # =================================
    # Cuisine → MultiLabel (N columns)
    # =================================
    cuisine_lists = (
        df["cuisine"]
        .astype(str)
        .str.split(",")
        .apply(lambda x: [i.strip() for i in x])
        .tolist()
    )

    mlb = MultiLabelBinarizer()
    cuisine_encoded = mlb.fit_transform(cuisine_lists).astype(np.float32)

    # =================================
    # Combine (3 + 1 + N)
    # =================================
    final_matrix = np.hstack([
        numeric,
        city_encoded,
        cuisine_encoded
    ])

    print("Final encoded shape:", final_matrix.shape)
    print("Features = 3 numeric + 1 city +", len(mlb.classes_), "cuisines")

    # save
    np.save(
        r"C:\Users\sojay\DS\VS_Code\Project_4\data\encoded_data.npy",
        final_matrix
    )

    with open(
        r"C:\Users\sojay\DS\VS_Code\Project_4\models\encoder.pkl",
        "wb"
    ) as f:
        pickle.dump((city_encoder, mlb), f)

    return final_matrix

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    df = clean_data()
    encoded = encode_data(df)

    print("Encoded shape:", encoded.shape)
    print("✅ Preprocessing complete")
