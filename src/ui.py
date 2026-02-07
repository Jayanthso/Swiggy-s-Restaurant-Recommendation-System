import streamlit as st
from model_rec import Recommender

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Restaurant Recommender",
    page_icon="🍽",
    layout="wide"
)

st.title("🍽 Restaurant Recommendation System")


# -------------------------------------------------
# Load model only once
# -------------------------------------------------
@st.cache_resource
def load_model():
    return Recommender()

rec = load_model()


# -------------------------------------------------
# Sidebar Filters (DEFINE EVERYTHING HERE)
# -------------------------------------------------
with st.sidebar:
    st.header("🔎 Filters")

    cities = sorted(rec.clean_df["city"].dropna().astype(str).unique())
    cuisines = sorted(rec.clean_df["cuisine"].dropna().astype(str).unique())

    city = st.selectbox("City", cities)
    cuisine = st.selectbox("Cuisine", cuisines)

    rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
    cost = st.slider("Budget (₹)", 50, 1000, 200)

    top_k = st.slider("Top Results", 5, 50, 10)

    sort_by = st.selectbox(
        "Sort by",
        ["rating", "cost"]
    )

    recommend_btn = st.button("🚀 Recommend")


# -------------------------------------------------
# Session state
# -------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None


# -------------------------------------------------
# Run only on button click
# -------------------------------------------------
if recommend_btn:
    with st.spinner("Finding best restaurants for you..."):

        result = rec.recommend_from_input(city, cuisine, rating, cost)

        result = (
            result
            .sort_values(sort_by, ascending=False)
            .head(top_k)
        )

        st.session_state.result = result


# -------------------------------------------------
# Display results
# -------------------------------------------------
if st.session_state.result is not None:

    df = st.session_state.result.copy()

    st.subheader("⭐ Top Recommendations")

    # Search
    search = st.text_input("Search restaurant name")

    if search:
        df = df[df["name"].str.contains(search, case=False)]

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Results", len(df))
    c2.metric("Avg Rating", round(df["rating"].mean(), 2))
    c3.metric("Avg Cost", f"₹{int(df['cost'].mean())}")

    # Formatting
    df_display = df.copy()
    df_display["rating"] = df_display["rating"].apply(lambda x: f"⭐ {x}")
    df_display["cost"] = df_display["cost"].apply(lambda x: f"₹ {x}")

    st.dataframe(
        df_display[["name", "rating", "cost", "cuisine", "address"]],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇ Download CSV",
        df.to_csv(index=False),
        file_name="restaurant_recommendations.csv",
        mime="text/csv"
    )

else:
    st.info("Select filters and click **Recommend** to see suggestions.")
