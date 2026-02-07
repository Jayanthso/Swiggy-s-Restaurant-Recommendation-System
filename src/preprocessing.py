import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import OneHotEncoder


RAW_PATH = r"C:\Users\sojay\DS\VS_Code\Project_4\data\swiggy.csv"


# =========================================================
# CLEAN DATA
# =========================================================
def clean_data():

    df = pd.read_csv(RAW_PATH)

    # -------------------------
    # Remove duplicates
    # -------------------------
    df = df.drop_duplicates()

    # -------------------------
    # Rating cleaning
    # -------------------------
    df["rating"] = df["rating"].replace("--", np.nan)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # -------------------------
    # rating_count -> numeric
    # "100+ ratings" -> 100
    # "Too Few Ratings" -> 0
    # -------------------------
    def extract_count(x):
        if isinstance(x, str):
            nums = re.findall(r"\d+", x)
            return int(nums[0]) if nums else 0
        return 0

    df["rating_count"] = df["rating_count"].apply(extract_count)
    df["rating_count"] = df["rating_count"].fillna(0).astype(int)

    # -------------------------
    # Cost cleaning
    # -------------------------
    df["cost"] = (
        df["cost"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
    )

    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["cost"] = df["cost"].fillna(df["cost"].median()).astype(int)

    # -------------------------
    # Reset index (important for mapping)
    # -------------------------
    df.reset_index(drop=True, inplace=True)

    # Save cleaned dataset
    df.to_csv(
        r"C:\Users\sojay\DS\VS_Code\Project_4\data\cleaned_data.csv",
        index=False
    )

    return df


# =========================================================
# ENCODE DATA
# =========================================================
def encode_data(df):

    categorical_cols = ["city", "cuisine"]
    numerical_cols = ["rating", "rating_count", "cost"]

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded_cat = encoder.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(encoded_cat)

    final_df = pd.concat(
        [df[numerical_cols].reset_index(drop=True), encoded_df],
        axis=1
    )

    final_df.to_csv(
        r"C:\Users\sojay\DS\VS_Code\Project_4\data\encoded_data.csv",
        index=False
    )

    # save encoder
    with open(
        r"C:\Users\sojay\DS\VS_Code\Project_4\models\encoder.pkl",
        "wb"
    ) as f:
        pickle.dump(encoder, f)

    return final_df


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    df = clean_data()
    encode_data(df)

    print("✅ Preprocessing complete (no similarity matrix built)")
