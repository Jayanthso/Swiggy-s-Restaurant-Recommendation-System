import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt


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

    # ------------------------------
    # Metric helper
    # ------------------------------
    @staticmethod
    def precision_at_k(recommended, relevant, k=10):
        recommended_k = recommended[:k]
        return len(set(recommended_k) & set(relevant)) / k

    # ------------------------------
    # Core similarity helper
    # ------------------------------
    def recommend_from_input(self, city, cuisine, rating, cost, top_k=10):

        cuisine_clean = str(cuisine).strip().lower()
    
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
    
        if len(candidate_idx) == 0:
            return self.clean_df.iloc[0:0].copy()
    
        numeric = np.array([[rating, 0, cost]], dtype=np.float32)
        city_encoded = self.city_encoder.transform([city]).reshape(1, -1)
    
        if cuisine_clean == "any":
            cuisine_encoded = np.zeros((1, len(self.mlb.classes_)), dtype=np.float32)
        else:
            cuisine_encoded = self.mlb.transform([[cuisine]]).astype(np.float32)
    
        user_vector = np.hstack([numeric, city_encoded, cuisine_encoded]).astype(np.float32)
    
        subset_matrix = self.encoded_matrix[candidate_idx]
    
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(subset_matrix)
    
        k = min(top_k, len(subset_matrix))
        _, indices = nn.kneighbors(user_vector, n_neighbors=k)
    
        final_indices = candidate_idx[indices[0]]
    
        return self.clean_df.iloc[final_indices].copy()

    # ------------------------------
    # Cross-validation method
    # ------------------------------
    @staticmethod
    def recall_at_k(recommended, relevant, k=4):
        recommended_k = recommended[:k]
        return len(set(recommended_k) & set(relevant)) / len(relevant) if relevant else 0
    
    @staticmethod
    def ndcg_at_k(recommended, relevant, k=4):
        recommended_k = recommended[:k]
        dcg = sum([1/np.log2(i+2) for i, r in enumerate(recommended_k) if r in relevant])
        idcg = sum([1/np.log2(i+2) for i in range(min(len(relevant), k))])
        return dcg / idcg if idcg > 0 else 0


    def cross_validate(self, n_splits=3, sample_size=100, top_k=4):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        precisions, recalls, ndcgs = [], [], []
        X = self.clean_df.drop("name", axis=1)
        y = self.clean_df["name"]
        
        #for fold_num, (train_idx, test_idx) in enumerate(kf.split(self.clean_df), 1):
         #   train_matrix = self.encoded_matrix[train_idx]
          #  nn = NearestNeighbors(metric="cosine", algorithm="brute")
           # nn.fit(train_matrix)
        for fold_num, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


            fold_prec, fold_rec, fold_ndcg = [], [], []
            for idx in test_idx[:sample_size]:
                city = self.clean_df.iloc[idx]["city"]
                cuisine = self.clean_df.iloc[idx]["cuisine"]
                rating = self.clean_df.iloc[idx]["rating"]
                cost = self.clean_df.iloc[idx]["cost"]

                recs = self.recommend_from_input(city, cuisine, rating, cost, top_k=top_k)
                recommended = recs.index.tolist()

                relevant = self.clean_df[
                    (self.clean_df["city"] == city) &
                    (self.clean_df["cuisine"].str.contains(cuisine, case=False, na=False))
                ].index.tolist()

                fold_prec.append(self.precision_at_k(recommended, relevant, k=top_k))
                fold_rec.append(self.recall_at_k(recommended, relevant, k=top_k))
                fold_ndcg.append(self.ndcg_at_k(recommended, relevant, k=top_k))

            precisions.append(np.mean(fold_prec))
            recalls.append(np.mean(fold_rec))
            ndcgs.append(np.mean(fold_ndcg))

            print(f"Fold {fold_num}: Precision={precisions[-1]:.3f}, Recall={recalls[-1]:.3f}, NDCG={ndcgs[-1]:.3f}")

        print(f"\nAverage Precision@{top_k}: {np.mean(precisions):.4f}")
        print(f"Average Recall@{top_k}: {np.mean(recalls):.4f}")
        print(f"Average NDCG@{top_k}: {np.mean(ndcgs):.4f}")

        plt.plot(precisions, label="Precision")
        plt.plot(recalls, label="Recall")
        plt.plot(ndcgs, label="NDCG")
        plt.legend()
        plt.title("Cross-validation metrics per fold")
        plt.show()
