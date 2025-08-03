import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ✅ Set page config at the top
st.set_page_config(page_title="Shopper Spectrum", layout="centered")

# ------------------------------
# 🔁 Load Models and Data
# ------------------------------
@st.cache_resource
def load_models():
    similarity = np.load("similarity_matrix.npy")
    product_list = pickle.load(open("product_names.pkl", "rb"))
    product_mapping = pickle.load(open("product_mapping.pkl", "rb"))  # StockCode → Product Name
    kmeans = pickle.load(open("rfm_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return similarity, product_list, product_mapping, kmeans, scaler

similarity_matrix, product_list, product_mapping, kmeans_model, scaler_model = load_models()

# ------------------------------
# 🌐 Sidebar with Project Info
# ------------------------------
with st.sidebar:
    st.title("📌 About Shopper Spectrum")
    st.write("""
    **Features:**
    - Product Recommendations
    - Customer Segmentation (RFM)
    
    **How to use:**
    - Go to *Product Recommendation* tab → Enter a product code or name.
    - Go to *Customer Segmentation* tab → Enter RFM values.
    """)

# ------------------------------
# 🌐 App Title and Tabs
# ------------------------------
st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendations")

tab1, tab2 = st.tabs(["🔎 Product Recommendation", "👤 Customer Segmentation"])

# ------------------------------
# 🎯 Module 1: Product Recommendation
# ------------------------------
with tab1:
    st.markdown("### 📦 Enter a Product Code or Name")
    input_product = st.text_input("Type product code (exact match)", placeholder="e.g., 84029E")

    if st.button("Get Recommendations"):
        if input_product in product_list:
            sim_scores = list(enumerate(similarity_matrix[product_list.index(input_product)]))
            sorted_similar = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            st.markdown("#### 🧠 Top 5 Recommended Products")
            
            for i, (idx, score) in enumerate(sorted_similar, 1):
                product_name = product_mapping.get(product_list[idx], product_list[idx])
                st.markdown(f"""
                <div style="border:1px solid #ddd; padding:10px; border-radius:10px; margin-bottom:5px;">
                    <b>{i}. {product_name}</b><br>
                    Similarity Score: {round(score, 2)}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("⚠️ Product not found! Please check the code or try another.")

# ------------------------------
# 🎯 Module 2: Customer Segmentation
# ------------------------------
with tab2:
    st.markdown("### 📊 Enter Customer RFM Values")
    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (total purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total amount spent)", min_value=0.0, step=10.0)

    if st.button("Predict Cluster"):
        rfm_input = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        scaled_rfm = scaler_model.transform(rfm_input)
        cluster = kmeans_model.predict(scaled_rfm)[0]

        cluster_map = {
            0: "🔴 At-Risk",
            1: "🟢 High-Value",
            2: "🟡 Regular",
            3: "🔵 Occasional"
        }
        label = cluster_map.get(cluster, "Unknown Segment")

        # Color-coded output
        color_map = {
            "🔴 At-Risk": "red",
            "🟢 High-Value": "green",
            "🟡 Regular": "orange",
            "🔵 Occasional": "blue"
        }
        color = color_map[label]

        st.markdown(f"<h4 style='color:{color};'>Predicted Segment: {label}</h4>", unsafe_allow_html=True)
