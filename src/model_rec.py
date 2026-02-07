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

        self.encoded_df = pd.read_csv(
            r"C:\Users\sojay\DS\VS_Code\Project_4\data\encoded_data.csv"
        )

        # smaller memory
        self.encoded_matrix = self.encoded_df.values.astype(np.float32)

        with open(r"C:\Users\sojay\DS\VS_Code\Project_4\models\encoder.pkl", "rb") as f:
            self.encoder = pickle.load(f)

        # faster than cosine_similarity
        self.nn = NearestNeighbors(metric="cosine")
        self.nn.fit(self.encoded_matrix)

        print("Recommender ready")

    # ------------------------------------------
    # By index
    # ------------------------------------------
    def recommend_by_index(self, idx, top_n=5):

        query = self.encoded_matrix[idx].reshape(1, -1)

        dist, indices = self.nn.kneighbors(query, n_neighbors=top_n+1)

        return self.clean_df.iloc[indices[0][1:]]

    # ------------------------------------------
    # By user input
    # ------------------------------------------
    def recommend_from_input(self, city, cuisine, rating, cost, top_n=5):

        input_df = pd.DataFrame({
            "city": [city],
            "cuisine": [cuisine]
        })

        encoded_cat = self.encoder.transform(input_df)

        numeric = np.array([[rating, 0, cost]], dtype=np.float32)

        user_vector = np.concatenate([numeric, encoded_cat], axis=1)

        dist, indices = self.nn.kneighbors(user_vector, n_neighbors=top_n)

        return self.clean_df.iloc[indices[0]]
