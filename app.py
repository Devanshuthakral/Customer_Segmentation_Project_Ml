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
st.markdown("### Analyze customer behavior using Machine Learning Clustering")

# Load models
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
kmeans = pickle.load(open("kmeans.pkl", "rb"))

# Load dataset
df = pd.read_csv("train.csv")

# Sidebar
st.sidebar.header("⚙️ Controls")
show_data = st.sidebar.checkbox("Show Raw Data", True)

# Feature Engineering (Demo purpose)
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

# 🔥 Human-friendly labels
cluster_labels = {
    0: "💰 Budget Customers",
    1: "👑 Premium Customers",
    2: "📱 Regular Users"
}
df['Customer Segment'] = df['Cluster'].map(cluster_labels)

# ===========================
# 📊 Metrics Section
# ===========================
col1, col2, col3 = st.columns(3)

col1.metric("👥 Total Customers", len(df))
col2.metric("🧩 Customer Segments", len(set(clusters)))

score = silhouette_score(X_scaled, clusters)
col3.metric("📈 Model Score", f"{score:.3f}")

# ===========================
# 📊 Dataset Preview
# ===========================
if show_data:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

# ===========================
# 📍 PCA Visualization
# ===========================
st.subheader("📍 Customer Segments Visualization")

fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis')

ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title("Customer Segmentation using K-Means")

# Legend for non-technical users
for i in np.unique(clusters):
    ax.scatter([], [], label=cluster_labels[i])
ax.legend(title="Customer Segment")

st.pyplot(fig)

# ===========================
# 📊 Distribution
# ===========================
st.subheader("📊 Customer Segment Distribution")
st.bar_chart(df['Customer Segment'].value_counts())

# ===========================
# 📈 Analysis Table
# ===========================
st.subheader("📈 Segment-wise Customer Analysis")
st.dataframe(df.groupby('Customer Segment')[features].mean())

# ===========================
# 🧠 Business Insights
# ===========================
st.subheader("🧠 Business Insights")

col1, col2, col3 = st.columns(3)

col1.info("""
💰 **Budget Customers**
- Spend less money  
- Highly price-sensitive  
👉 Strategy: Provide discounts and offers
""")

col2.success("""
👑 **Premium Customers**
- High spending users  
- Loyal and valuable customers  
👉 Strategy: Loyalty programs and exclusive perks
""")

col3.warning("""
📱 **Regular Users**
- Frequently use the app  
- Moderate spending behavior  
👉 Strategy: Personalized recommendations
""")

# ===========================
# Footer
# ===========================
st.markdown("---")
st.caption("🚀 Developed using Streamlit | Customer Segmentation ML Project")
