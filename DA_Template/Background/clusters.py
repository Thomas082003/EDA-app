import streamlit as st
from PIL import Image

def clusters_page():
    # Title for the page
    st.title("K-Means Clustering Visualizations")
    st.markdown(
        """
        Explore the clustering results for sentiment analysis using K-Means. The visualizations highlight 
        how the data is grouped into clusters based on different vectorization methods and feature reductions.
        """
    )

    # Load and display the third 2D clustering visualization (Word2Vec)
    st.markdown("### First Debate 2D")
    try:
        image_2d_word2vec = Image.open("images/KMeans_2D_Clusters_Word2Vec.jpeg")
        st.image(image_2d_word2vec, caption="K-Means Clustering in 2D - Word2Vec", use_column_width=True)
    except FileNotFoundError:
        st.error("KMeans_2D_Clusters_Word2Vec.jpeg not found.")

    # Load and display the third 3D clustering visualization (Word2Vec)
    st.markdown("### First Debate 3D")
    try:
        image_3d_word2vec = Image.open("images/KMeans_3D_Clusters_Word2Vec.jpeg")
        st.image(image_3d_word2vec, caption="K-Means Clustering in 3D - Word2Vec", use_column_width=True)
    except FileNotFoundError:
        st.error("KMeans_3D_Clusters_Word2Vec.jpeg not found.")

        # Load and display the second 2D clustering visualization (TF-IDF)
    st.markdown("### Second Debate 2D")
    try:
        image_2d_tfidf = Image.open("images/KMeans_2D_Clusters_TF-IDF.jpeg")
        st.image(image_2d_tfidf, caption="K-Means Clustering in 2D - TF-IDF", use_column_width=True)
    except FileNotFoundError:
        st.error("KMeans_2D_Clusters_TF-IDF.jpeg not found.")

        # Load and display the second 3D clustering visualization (TF-IDF)
    st.markdown("### Second Debate 3D")
    try:
        image_3d_tfidf = Image.open("images/KMeans_3D_Clusters_TF-IDF.jpeg")
        st.image(image_3d_tfidf, caption="K-Means Clustering in 3D - TF-IDF", use_column_width=True)
    except FileNotFoundError:
        st.error("KMeans_3D_Clusters_TF-IDF.jpeg not found.")

    # Load and display the first 2D clustering visualization
    st.markdown("### Third Debate 2D")
    try:
        image_2d_full = Image.open("images/KMeans_2D_Clusters_Full.jpeg")
        st.image(image_2d_full, caption="K-Means Clustering in 2D - Full Dataset", use_column_width=True)
    except FileNotFoundError:
        st.error("KMeans_2D_Clusters_Full.jpeg not found.")


    # Load and display the first 3D clustering visualization
    st.markdown("### Third Debate 3D")
    try:
        image_3d_full = Image.open("images/KMeans_3D_Clusters_Full.jpeg")
        st.image(image_3d_full, caption="K-Means Clustering in 3D - Full Dataset", use_column_width=True)
    except FileNotFoundError:
        st.error("KMeans_3D_Clusters_Full.jpeg not found.")



