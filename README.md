# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendation in E-Commerce

Welcome to **Shopper Spectrum**, a machine learning-powered analytics tool designed to enhance e-commerce strategies through **RFM-based customer segmentation** and **product recommendations**. Built as part of an internship project at **Labmentix**, this application helps identify customer patterns and deliver personalized shopping experiences.

---

## 📌 Project Highlights

- 🎯 **Customer Segmentation** using RFM (Recency, Frequency, Monetary) analysis + KMeans Clustering
- 🤖 **Product Recommendation System** using Item-based Collaborative Filtering (Cosine Similarity)
- 🧮 **Machine Learning Models** built with Scikit-learn, visualized via Streamlit
- 🌐 **Interactive Web App** using Streamlit UI with real-time inputs

---

## 🧠 Technologies Used

- Python, Pandas, NumPy
- Scikit-learn (KMeans, StandardScaler)
- Streamlit (Frontend Deployment)
- Cosine Similarity (Collaborative Filtering)
- Data Visualization: Seaborn, Matplotlib

---

## 🚀 Features

### 1. 📦 Product Recommendation
- Input a **Product Code**
- Get **Top 5 Similar Products**
- Based on purchase behavior and similarity matrix

### 2. 👤 Customer Segmentation
- Input: **Recency**, **Frequency**, **Monetary**
- Output: Predicted customer segment:
  - 🟢 High-Value
  - 🟡 Regular
  - 🔵 Occasional
  - 🔴 At-Risk

---

## 🗂️ Folder Structure

├── shop_spectrum.ipynb # Main Jupyter Notebook
├── streamlit_shop.py # Streamlit App
├── online_retail.csv # Dataset
├── similarity_matrix.npy # Recommendation matrix
├── product_names.pkl # Product ID list
├── product_mapping.pkl # Product Code → Name map
├── rfm_model.pkl # Trained KMeans model
├── scaler.pkl # StandardScaler object
└── README.md # Project documentation


📊 Business Use Cases
🎯 Targeted Marketing Campaigns

🛍️ Personalized Shopping Experience

⚠️ Customer Retention (At-Risk detection)

📦 Inventory Optimization

📈 Data-Driven Strategy for Growth



📬 Contact
📧 Email: your.email@example.com
