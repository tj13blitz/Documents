import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
import altair as alt

df = pd.read_csv('/mount/src/documents/streamlit/stats.csv')
df.columns = ['Last Name, First Name', 'Player ID','Year','IP','Plate Appearances', 'Hits','Home Runs','K','BB','K%','BB%','OBP', 'BABIP','ERA','HBP','Soft Contact%','Hard Hit%','Whiff%','GB%','FB%','LD%','Popup%','TP','Fastball%','Breaking%','Offspeed%']

# Create calculated stats
FIP = ((13 * df['Home Runs']) + (3 * (df['BB'] + df['HBP'])) - (2 * df['K'])) / (df['IP']) + 3.2
K_per_nine = df['K'] / 9
WHIP = (df['BB'] + df['Hits']) / df['IP']
TP_Per_9 = (df['TP'] / df['IP']) * 9

df['FIP'] = FIP
df['K/9'] = K_per_nine
df['WHIP'] = WHIP
df['TP/9'] = TP_Per_9

st.header("Data Science Capstone: MLB Pitchers, Contact vs. Strikeout")


st.write("Take a look at the data used! All 2024 pitchers with min. 100 PA.")
st.write("Choose which stats you want to see or search for specific players.")

# Create multiselect bar to show/hide different columns
with st.container():
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display:",
        options=all_columns,
        default=all_columns,
    )

# Display the dataframe
with st.container():
    st.dataframe(df[selected_columns])

#Select Features
features = ["K%", "BB%", "BABIP", "Soft Contact%", "Hard Hit%",
       "Whiff%", "GB%", "FB%", "LD%", "Popup%", "Fastball%", "Breaking%",
       "Offspeed%", "K/9", "TP/9"]

#Convert any NaN to mean value
imputer = SimpleImputer(strategy='mean')
df[features] = imputer.fit_transform(df[features])

#Scale Data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

#Find optimal k value (Elbow Method)
inertia = []
for k in range(1,10):
  kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
  kmeans.fit(scaled_features)
  inertia.append(kmeans.inertia_)

# Create elbow data dataframe to use in streamlit line chart
elbow_data = pd.DataFrame({
    'Number of Clusters (k)': range(1, 10),
    'Inertia': inertia
})

st.header("Elbow Analysis Reveals 3 Clusters as Optimal")

with st.container():
    chart = alt.Chart(elbow_data).mark_line().encode(
        x='Number of Clusters (k)',
        y=alt.Y('Inertia', scale=alt.Scale(domain=[8200, 5000], reverse=True)),
        tooltip=['Number of Clusters (k)', 'Inertia']
    ).properties(
        width='container'
    )
    
    st.altair_chart(chart, use_container_width=True)

#Apply K-Means for k=3
kmeans_3 = KMeans(n_clusters=3, random_state=42)
clusters_3 = kmeans_3.fit_predict(scaled_features)

#Analyze clusters
cluster_mapping = {
    0: "Strikeout",  # Map cluster 0 to Group A
    1: "Contact",  # Map cluster 1 to Group B
    2: "Hybrid"   # Map cluster 2 to Group C
}
cluster_color = {
    "Strikeout": "red",
    "Contact": "green",
    "Hybrid": "blue"
}

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Cluster Analysis Visualizations", key="cluster_button"):
        # This will toggle the state between True and False
        st.session_state.show_df = not st.session_state.get('show_df', False)

# Initialize the state
if 'show_df' not in st.session_state:
    st.session_state.show_df = False

if st.session_state.show_df:
    df["Cluster"] = clusters_3
    df['Group'] = df['Cluster'].map(cluster_mapping)
    st.header("Mean Values for Each Feature by Cluster")
    st.dataframe(df.groupby("Group")[features].mean())
    st.write("This shows the clear distinctions between the features the groups were clustered by.")

    cluster_stats = df.groupby('Group')[['ERA', 'FIP', 'WHIP', 'BABIP']].agg(['mean', 'median'])
    st.header("Mean and Median Values of ERA, FIP, WHIP, and BABIP by Cluster")
    st.dataframe(cluster_stats)
    st.write("This shows that strikeout pitchers are the most effective in terms of ERA, FIP and WHIP, while contact pitchers are the most effective in terms of BABIP. This makes sense as the traditional contact pitcher generates more weak contact.")

# Create a button to toggle content
with col2:
    if st.button("Cluster Analysis Visualizations"):
        # This will toggle the state between True and False
        st.session_state.show_cluster_analysis = not st.session_state.get('show_cluster_analysis', False)

# Initialize the state
if 'show_cluster_analysis' not in st.session_state:
    st.session_state.show_cluster_analysis = False

if st.session_state.show_cluster_analysis:
    st.header("Pitcher Clusters, WHIP vs. ERA")
    fig = px.scatter(
        df,
        x='WHIP',
        y='ERA',
        color='Group',
        color_discrete_map=cluster_color,
        title=" ",
        height=600,
        hover_data=['Last Name, First Name', 'Group', 'BABIP', 'FIP']  # Show all columns in hover
    )

    # Customize layout
    fig.update_layout(
        title={'y':0.95, 'x':0.5},
        xaxis_title="WHIP",
        yaxis_title="ERA",
        legend_title="Cluster Group"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display in Streamlit
    st.write("This shows the different clusters and how individuals perform based on ERA and WHIP, strikeout pitchers tend to be in the lower left corner making them the most effective in these aspects.")

    st.header("Pitcher Clusters, K% vs. GB%")
    fig2 = px.scatter(
        df,
        x='K%',
        y='GB%',
        color='Group',
        color_discrete_map=cluster_color,
        title=" ",
        height=600,
        hover_data=['Last Name, First Name', 'Group', 'BABIP', 'FIP', 'ERA', 'WHIP']  # Show all columns in hover
    )

    # Customize layout
    fig2.update_layout(
        title={'y':0.95, 'x':0.5},
        xaxis_title="K%",
        yaxis_title="GB%",
        legend_title="Cluster Group"
    )

    # Display in Streamlit
    st.plotly_chart(fig2, use_container_width=True)
    st.write("This graph shows how the clusters line up clearly with their titles, strikeout pitchers with a high K% and contact pitchers with a high GB%, with hybrid pitchers falling in the middle.")

    st.header("Pitcher Clusters, K% vs. BABIP")
    fig3 = px.scatter(
        df,
        x='K%',
        y='GB%',
        color='Group',
        color_discrete_map=cluster_color,
        title=" ",
        height=600,
        hover_data=['Last Name, First Name', 'Group', 'BABIP', 'FIP', 'ERA', 'WHIP', 'K%']  # Show all columns in hover
    )

    # Customize layout
    fig3.update_layout(
        title={'y':0.95, 'x':0.5},
        xaxis_title="K%",
        yaxis_title="BABIP",
        legend_title="Cluster Group"
    )

    # Display in Streamlit
    st.plotly_chart(fig3, use_container_width=True)
    st.write("This graph shows how the clusters line up clearly with their titles, strikeout pitchers have a high K% and contact pitchers have a high average on balls put in play.")

# Select numerical features for PCA
pca_features = ['K%', 'BB%', 'BABIP', 'Soft Contact%', 'Hard Hit%', 'Whiff%', 'GB%', 'FB%', 'LD%', 'Popup%', 'Fastball%', 'Breaking%', 'Offspeed%', 'K/9', 'TP/9']

# Impute missing values (if any) before PCA
imputer = SimpleImputer(strategy='mean')
df[pca_features] = imputer.fit_transform(df[pca_features])

# Scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[pca_features])

# Apply PCA
pca = PCA(n_components=3)  # Reduce to 2 principal components
principal_components = pca.fit_transform(scaled_data)

# Create a new DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

principal_df['Group'] = df['Group']
principal_df['Last Name, First Name'] = df['Last Name, First Name']
principal_df['BABIP'] = df['BABIP']
principal_df['FIP'] = df['FIP']
principal_df['ERA'] = df['ERA']
principal_df['WHIP'] = df['WHIP']
principal_df['K%'] = df['K%']

# Create a button to toggle content
with col3:
    if st.button("PCA Visualizations", key="PCA_button"):
        # This will toggle the state between True and False
        st.session_state.show_pca = not st.session_state.get('show_pca', False)

# Initialize the state if it doesn't exist
if 'show_pca' not in st.session_state:
    st.session_state.show_pca = False

if st.session_state.show_pca:
    fig4 = px.scatter(
        principal_df, 
        x='PC1',
        y='PC2',
        color='Group',
        color_discrete_map=cluster_color,
        title="PCA of Pitcher Data (PC1 vs PC2)",
        height=600,
        hover_data=['Last Name, First Name', 'Group', 'BABIP', 'FIP', 'ERA', 'WHIP', 'K%']
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.write("The principal component analysis shows how principal components 1 and 2 separate the 3 groups of pitchers fairly well.")

    fig5 = px.scatter(
        principal_df, 
        x='PC1',
        y='PC3',
        color='Group',
        color_discrete_map=cluster_color,
        title="PCA of Pitcher Data (PC1 vs PC3)",
        height=600,
        hover_data=['Last Name, First Name', 'Group', 'BABIP', 'FIP', 'ERA', 'WHIP', 'K%']
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.write("The principal component analysis shows how principal components 1 and 3 clearly separate the 3 groups of pitchers.")
