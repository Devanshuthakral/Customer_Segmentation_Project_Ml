import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import silhouette_score

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title
st.title("🍔 Customer Segmentation Dashboard")
st.markdown("### Analyze customer behavior using ML clustering")

# Load models
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
kmeans = pickle.load(open("kmeans.pkl", "rb"))

# Load dataset
df = pd.read_csv("train.csv")

# Sidebar
st.sidebar.header("⚙️ Controls")
show_data = st.sidebar.checkbox("Show Raw Data", True)

# Feature Engineering
np.random.seed(42)
df['Age'] = np.random.randint(18, 60, len(df))
df['TotalOrders'] = np.random.randint(1, 50, len(df))
df['AvgSpend'] = np.random.randint(100, 1000, len(df))
df['AppUsageTime'] = np.random.randint(5, 120, len(df))

features = ['Age', 'TotalOrders', 'AvgSpend', 'AppUsageTime']
X = df[features]

# Model prediction
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)
clusters = kmeans.predict(X_scaled)
df['Cluster'] = clusters

# Metrics row
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Clusters", len(set(clusters)))

score = silhouette_score(X_scaled, clusters)
col3.metric("Silhouette Score", f"{score:.3f}")

# Show dataset
if show_data:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

# Graph Section
st.subheader("📍 Customer Segmentation (PCA)")

fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis')
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("KMeans Clustering")
st.pyplot(fig)

# Cluster Distribution
st.subheader("📊 Cluster Distribution")
cluster_counts = df['Cluster'].value_counts()
st.bar_chart(cluster_counts)

# Cluster Insights
st.subheader("📈 Cluster Analysis")
st.dataframe(df.groupby('Cluster')[features].mean())

# Business Insights
st.subheader("🧠 Business Insights")

st.info("""
🔹 Cluster 0 → Low spenders → Offer discounts  
🔹 Cluster 1 → High spenders → Loyalty programs  
🔹 Cluster 2 → Frequent users → Personalized recommendations  
""")

# Footer
st.markdown("---")
st.caption("Developed using Streamlit | ML Project")