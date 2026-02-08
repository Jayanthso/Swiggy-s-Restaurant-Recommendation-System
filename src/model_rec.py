import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors


class Recommender:

    def __init__(self):

        print("Loading recommender...")

        self.clean_df = pd.read_csv(
            r"C:\Users\sojay\DS\VS_Code\Project_4\data\cleaned_data.csv"
        )

        self.encoded_matrix = np.load(
            r"C:\Users\sojay\DS\VS_Code\Project_4\data\encoded_data.npy"
        )

        with open(
            r"C:\Users\sojay\DS\VS_Code\Project_4\models\encoder.pkl", "rb"
        ) as f:
            self.city_encoder, self.mlb = pickle.load(f)

        # Fit once on full matrix
        self.nn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=-1
        )

        self.nn.fit(self.encoded_matrix)

        print("Recommender ready ✓")


    # ==================================================
    # Core similarity helper
    # ==================================================
    def recommend_from_input(self, city, cuisine, rating, cost, top_k=10):

        cuisine_clean = str(cuisine).strip().lower()
    
        # ----------------------------------
        # HARD FILTER (correct logic)
        # ----------------------------------
        mask = (
            (self.clean_df["city"] == city) &
            (self.clean_df["rating"] >= rating) &
            (self.clean_df["cost"] <= cost)
        )
    
        if cuisine_clean != "any":
            mask &= self.clean_df["cuisine"].str.contains(
                cuisine_clean, case=False, na=False
            )
    
        candidate_idx = np.where(mask)[0]
    
        print("Candidates after filter:", len(candidate_idx))
    
        if len(candidate_idx) == 0:
            return self.clean_df.iloc[0:0].copy()
    
        # ----------------------------------
        # build vector
        # ----------------------------------
        numeric = np.array([[rating, 0, cost]], dtype=np.float32)
    
        city_encoded = self.city_encoder.transform([city]).reshape(1, -1)
    
        if cuisine_clean == "any":
            cuisine_encoded = np.zeros((1, len(self.mlb.classes_)), dtype=np.float32)
        else:
            cuisine_encoded = self.mlb.transform([[cuisine]]).astype(np.float32)
    
        user_vector = np.hstack([numeric, city_encoded, cuisine_encoded]).astype(np.float32)
    
        # ----------------------------------
        # KNN
        # ----------------------------------
        subset_matrix = self.encoded_matrix[candidate_idx]
    
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(subset_matrix)
    
        k = min(top_k, len(subset_matrix))
    
        _, indices = nn.kneighbors(user_vector, n_neighbors=k)
    
        final_indices = candidate_idx[indices[0]]
    
        return self.clean_df.iloc[final_indices].copy()
    
    
