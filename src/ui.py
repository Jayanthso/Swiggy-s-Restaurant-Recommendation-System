import streamlit as st
from model_rec import Recommender

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Restaurant Recommender",
    page_icon="🍽",
    layout="wide"
)

st.title("🍽 Restaurant Recommendation System")


# -------------------------------------------------
# Load model once (cached)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return Recommender()

rec = load_model()


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("🔎 Filters")

    # -------------------------
    # City
    # -------------------------
    cities = sorted(rec.clean_df["city"].dropna().unique())
    city = st.selectbox("City", cities)

    # -------------------------
    # Cuisine (split properly)
    # -------------------------
    cuisine_set = set()
    for c in rec.clean_df["cuisine"].dropna():
        cuisine_set.update([i.strip() for i in str(c).split(",")])

    cuisines = ["Any"] + sorted(cuisine_set)

    cuisine = st.selectbox("Cuisine", cuisines)

    # -------------------------
    # sliders
    # -------------------------
    rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
    cost = st.slider("Budget (₹)", 50, 1000, 200)

    top_k = st.slider("Top Results", 5, 50, 10)

    sort_by = st.selectbox("Sort by", ["rating", "cost"])

    recommend_btn = st.button("🚀 Recommend")


# -------------------------------------------------
# Session state
# -------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None

# -------------------------------------------------
# Recommend
# -------------------------------------------------
if recommend_btn:

    with st.spinner("Finding restaurants..."):

        # -------------------------
        # PRE-FILTER by city (actual use)
        # -------------------------
        city_mask = rec.clean_df["city"] == city
        city_indices = rec.clean_df[city_mask].index

        if len(city_indices) == 0:
            st.warning("No restaurants found for this city")
            st.session_state.result = rec.clean_df.iloc[[]]

        else:
            result = rec.recommend_from_input(
                city, cuisine, rating, cost, top_k * 5
            )

            # -------------------------
            # Post filters
           
            # -------------------------
            # sorting
            # -------------------------
            ascending = (sort_by == "cost")

            result = (
                result
                .sort_values(sort_by, ascending=ascending)
                .head(top_k)
            )

            st.session_state.result = result


# -------------------------------------------------
# Display
# -------------------------------------------------
if st.session_state.result is not None:

    df = st.session_state.result.copy()

    if df.empty:
        st.warning("No restaurants match your filters.")
        st.stop()

    st.subheader("⭐ Top Recommendations")

    search = st.text_input("Search restaurant")

    if search:
        df = df[df["name"].str.contains(search, case=False, na=False)]

    if df.empty:
        st.warning("No results after search filter.")
        st.stop()

    c1, c2, c3 = st.columns(3)

    c1.metric("Results", len(df))
    c2.metric("Avg Rating", round(float(df["rating"].mean()), 2))
    c3.metric("Avg Cost", f"₹{int(df['cost'].mean())}")

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
        file_name="restaurant_recommendations.csv"
    )

else:
    st.info("Select filters and click **Recommend**.")
