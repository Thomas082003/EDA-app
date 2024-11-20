import streamlit as st
from PIL import Image
import pandas as pd

def sentiment_analysis_page():
    # Page title
    st.title("Sentiment Analysis Dashboard")
    # Class Data Distribution Plot
    st.markdown("#### Class Data Distribution")
    st.markdown(
        """
        This plot represents the distribution of the data across three sentiment classes: 
        - **0 (Negative)**: Sentiments categorized as negative.
        - **1 (Neutral)**: Sentiments that are neutral in tone.
        - **2 (Positive)**: Sentiments categorized as positive.

        By visualizing the distribution of sentiment classes, we can analyze the proportion of each sentiment and identify any potential patterns or imbalances in the data.
        """
    )
    try:
        # Load and display the image
        class_distribution_img = Image.open("images/class_data_distribution.jpeg")
        st.image(class_distribution_img, caption="Class Data Distribution", use_column_width=True)
    except FileNotFoundError:
        st.error("Class Data Distribution image not found. Please check the file path.")

    # Training and Test Distribution Plot
    st.markdown("#### Training and Test Distribution Classes")
    st.markdown(
        """
        This plot shows the number of samples in each sentiment class for both the training and test datasets:
        - **Training Dataset**: Used to train sentiment models.
        - **Test Dataset**: Used to evaluate model performance.

        Maintaining a consistent distribution between these datasets ensures fair and unbiased evaluation. 
        """
    )
    try:
        # Load and display the image
        training_test_distribution_img = Image.open("images/training_test_distribution.jpeg")
        st.image(training_test_distribution_img, caption="Training and Test Distribution Classes", use_column_width=True)
    except FileNotFoundError:
        st.error("Training and Test Distribution Classes image not found. Please check the file path.")

    # Additional Information Section
    st.markdown("### Overview of Vectorization Techniques")
    st.markdown(
        """
        To enable sentiment analysis models to work effectively with numerical data, we used two vectorization techniques:
        
        **1. TF-IDF (Term Frequency-Inverse Document Frequency):**
        - A statistical measure used to determine the importance of words in documents.
        - Words are represented as individual features in a sparse vector with a length equal to the vocabulary size.
        - Focuses on the importance of words in a document relative to the entire corpus.
        - Best used when the statistical importance of words is sufficient, and understanding context is less critical.

        **2. Word2Vec:**
        - Creates dense vectors (embeddings) for words in a continuous vector space.
        - Captures semantic relationships; similar words have similar vector representations.
        - Scans the entire corpus to identify relationships between words based on co-occurrence.
        - Best used when capturing semantic relationships and understanding context is essential.

        ### Process Summary
        - The dataset is split into **training (90%)** and **testing (10%)** sets.
        - Three model versions are trained and evaluated using both vectorization methods.
        - The best-performing model and vectorization method will be selected based on performance metrics.
        
        By following this approach, we ensure the models generalize effectively to new, unseen data, allowing for a more accurate performance assessment.
        """
    )

  # Word2Vec Versions Tables
    st.markdown("### Word2Vec Model Comparisons")
    st.markdown(
        """
        Below are the results for three versions of Word2Vec-based models. Each table summarizes the performance metrics 
        for different algorithms using the Word2Vec vectorization technique.
        """
    )

    # Version 1 Table
    st.markdown("#### Version 1 Results")
    version1_data = {
        "Metrics": ["Accuracy", "Recall", "Precision", "F1 Score", "ROC AUC (Negative)", "ROC AUC (Neutral)", "ROC AUC (Positive)", "Overall ROC AUC"],
        "Naive Bayes": [0.454, 0.478, 0.552, 0.442, 0.656, 0.671, 0.674, 0.667],
        "KNN (3 Neighbors)": [0.473, 0.458, 0.478, 0.457, 0.617, 0.630, 0.639, 0.629],
        "CART (Gini)": [0.464, 0.458, 0.459, 0.448, 0.561, 0.603, 0.586, 0.578],
        "CART (Entropy)": [0.471, 0.459, 0.459, 0.473, 0.600, 0.572, 0.580, 0.588],
        "MLP (64,32,16)": [0.550, 0.522, 0.582, 0.528, 0.697, 0.700, 0.699, 0.716],
        "SVM (RBF Kernel)": [0.449, 0.372, 0.476, 0.271, 0.686, 0.630, 0.657, 0.657],
        "XGBoost Classifier": [0.579, 0.561, 0.588, 0.567, 0.716, 0.741, 0.739, 0.732],
    }
    st.table(version1_data)

    # Version 2 Table
    st.markdown("#### Version 2 Results")
    version2_data = {
        "Metrics": ["Accuracy", "Recall", "Precision", "F1 Score", "ROC AUC (Negative)", "ROC AUC (Neutral)", "ROC AUC (Positive)", "Overall ROC AUC"],
        "Naive Bayes": [0.454, 0.478, 0.552, 0.442, 0.656, 0.671, 0.674, 0.667],
        "KNN (3 Neighbors)": [0.513, 0.498, 0.513, 0.482, 0.690, 0.680, 0.690, 0.676],
        "CART (Gini)": [0.510, 0.454, 0.456, 0.451, 0.586, 0.608, 0.590, 0.627],
        "CART (Entropy)": [0.473, 0.439, 0.439, 0.451, 0.579, 0.553, 0.573, 0.605],
        "MLP (64,32,16)": [0.541, 0.500, 0.556, 0.503, 0.700, 0.702, 0.703, 0.694],
        "SVM (Linear Kernel)": [0.466, 0.414, 0.505, 0.380, 0.600, 0.651, 0.661, 0.710],
        "XGBoost Classifier": [0.558, 0.500, 0.561, 0.542, 0.723, 0.728, 0.738, 0.710],
    }
    st.table(version2_data)

    # Version 3 Table
    st.markdown("#### Version 3 Results")
    version3_data = {
        "Metrics": ["Accuracy", "Recall", "Precision", "F1 Score", "ROC AUC (Negative)", "ROC AUC (Neutral)", "ROC AUC (Positive)", "Overall ROC AUC"],
        "Naive Bayes": [0.454, 0.478, 0.552, 0.442, 0.656, 0.671, 0.674, 0.667],
        "KNN (7 Neighbors)": [0.528, 0.516, 0.541, 0.516, 0.665, 0.694, 0.706, 0.688],
        "CART (Gini)": [0.501, 0.491, 0.500, 0.493, 0.596, 0.604, 0.659, 0.620],
        "CART (Entropy)": [0.471, 0.451, 0.462, 0.452, 0.587, 0.595, 0.618, 0.600],
        "MLP (50,8)": [0.415, 0.330, 0.138, 0.195, 0.453, 0.577, 0.505, 0.511],
        "SVM (Polynomial Kernel)": [0.520, 0.460, 0.656, 0.433, 0.703, 0.710, 0.696, 0.700],
        "XGBoost Classifier": [0.558, 0.538, 0.561, 0.542, 0.723, 0.728, 0.738, 0.730],
    }
    st.table(version3_data)

    # Best Models
    st.markdown("#### Best Models")
    best_models_data = {
        "Metrics": ["Accuracy", "Recall", "Precision", "F1 Score", "ROC AUC (Negative)", "ROC AUC (Neutral)", "ROC AUC (Positive)", "Overall ROC AUC"],
        "XGBoost with Depth 6": [0.575, 0.555, 0.595, 0.561, 0.718, 0.721, 0.721, 0.724],
        "SVM with Polynomial Kernel (Degree 3)": [0.494, 0.434, 0.596, 0.402, 0.695, 0.711, 0.666, 0.691],
        "KNN (7 Neighbors)": [0.486, 0.479, 0.500, 0.480, 0.637, 0.682, 0.649, 0.656],
    }
    st.table(best_models_data)

        # Interpretation of the ROC Curve
    st.markdown("### ROC Curve Analysis")
    st.markdown(
        """
        The ROC curve compares the performance of the three best models:  
        - **XGBoost (max_depth=6)**: Represented by the solid blue line.  
        - **KNN (K=7)**: Represented by the dashed orange line.  
        - **SVM (Polynomial Kernel, degree=3)**: Represented by the dashed green line with dots.  

        #### Key Observations:
        - **XGBoost** shows the highest ROC-AUC score (0.724), indicating a superior balance between sensitivity (True Positive Rate) and specificity (False Positive Rate).
        - **KNN (K=7)** achieves a moderate balance, performing slightly worse than XGBoost in some regions.
        - **SVM (Polynomial Kernel, degree=3)** provides competitive performance in specific ranges but falls behind XGBoost in terms of overall ROC-AUC.  

        The ROC curve highlights the ability of the models to distinguish between classes, with XGBoost emerging as the best performer.
        """
    )

    # Load and display the ROC curve image
    try:
        roc_curve_img = Image.open("images/roc_curve_best_models.jpeg")
        st.image(roc_curve_img, caption="ROC Curve of the Top 3 Models", use_column_width=True)
    except FileNotFoundError:
        st.error("ROC Curve image not found. Please check the file path.")

    # TF-IDF Vectorization Section
    st.markdown("### TF-IDF Vectorization Model Comparisons")
    st.markdown(
        """
        **TF-IDF Vectorization Model**: Statistical and sparse representation, ideal for cases where the statistical importance of words is sufficient, and understanding the context is less crucial.  

        We applied the same methodology to create the three versions of the model, with the key difference being the use of the TF-IDF vectorization technique. Below is the implementation outline:
        - **TF-IDF**: “Statistical measure used to determine the mathematical significance of words in documents.”
          - Each word is represented as an individual feature.
          - Results in a sparse vector representation where the length equals the size of the vocabulary.
          - Focuses on the importance of words in a document relative to the entire corpus.

        """
    )


    st.markdown("### Version 1 Results")
    version_1_data = {
        "Metrics": [
            "Accuracy", "Recall", "Precision", "F1-Score", 
            "ROC AUC for class Negative", "ROC AUC for class Neutral", 
            "ROC AUC for class Positive", "ROC-AUC"
        ],
        "Naïve Bayes Multinomial": [0.633, 0.603, 0.624, 0.595, 0.822, 0.76, 0.858, 0.813],
        "KNN 3 Neighbors": [0.541, 0.507, 0.56, 0.51, 0.71, 0.657, 0.754, 0.707],
        "CART Gini": [0.586, 0.588, 0.589, 0.586, 0.712, 0.678, 0.699, 0.696],
        "CART Entropy": [0.556, 0.552, 0.555, 0.552, 0.704, 0.629, 0.67, 0.668],
        "MLP 'lbfgs' solver, layers(64,32,16), relu activation": [0.603, 0.606, 0.602, 0.603, 0.737, 0.75, 0.802, 0.764],
        "SVM with RBF kernel. C:0.1, 'auto' gamma": [0.415, 0.333, 0.138, 0.195, 0.792, 0.754, 0.872, 0.806],
        "XGBoost Classifier": [0.631, 0.604, 0.622, 0.593, 0.811, 0.775, 0.871, 0.819]
    }
    st.table(version_1_data)

    st.markdown("### Version 2 Results")
    version_2_data = {
        "Metrics": [
            "Accuracy", "Recall", "Precision", "F1-Score", 
            "ROC AUC for class Negative", "ROC AUC for class Neutral", 
            "ROC AUC for class Positive", "ROC-AUC"
        ],
        "Naïve Bayes Multinomial": [0.633, 0.603, 0.624, 0.595, 0.822, 0.76, 0.858, 0.813],
        "KNN 5 Neighbors": [0.579, 0.566, 0.597, 0.562, 0.73, 0.72, 0.768, 0.739],
        "CART Gini with depth 12": [0.571, 0.529, 0.578, 0.17, 0.677, 0.629, 0.719, 0.675],
        "CART Entropy with depth 12": [0.571, 0.522, 0.6, 0.507, 0.681, 0.632, 0.719, 0.678],
        "MLP 'lbfgs' solver, layers(50,8), logistic activation": [0.648, 0.65, 0.647, 0.785, 0.763, 0.848, 0.848, 0.799],
        "SVM with linear kernel. C:0.1, 'auto' gamma": [0.643, 0.614, 0.636, 0.602, 0.832, 0.773, 0.882, 0.829],
        "XGBoost Classifier with depth 6": [0.62, 0.585, 0.622, 0.574, 0.798, 0.766, 0.868, 0.81]
    }
    st.table(version_2_data)

    st.markdown("### Version 3 Results")
    version_3_data = {
        "Metrics": [
            "Accuracy", "Recall", "Precision", "F1-Score", 
            "ROC AUC for class Negative", "ROC AUC for class Neutral", 
            "ROC AUC for class Positive", "ROC-AUC"
        ],
        "Naïve Bayes Multinomial": [0.633, 0.603, 0.624, 0.595, 0.822, 0.76, 0.858, 0.813],
        "KNN 7 Neighbors": [0.582, 0.567, 0.567, 0.564, 0.756, 0.722, 0.765, 0.748],
        "CART Gini with depth 12": [0.582, 0.538, 0.592, 0.515, 0.688, 0.63, 0.735, 0.686],
        "CART Entropy with depth 12": [0.569, 0.519, 0.596, 0.503, 0.675, 0.625, 0.731, 0.677],
        "MLP 'sgd' solver, layers(50,8), relu activation": [0.639, 0.633, 0.632, 0.606, 0.825, 0.782, 0.879, 0.829],
        "SVM with polynomial kernel. C:0.7, degree of 3": [0.626, 0.632, 0.592, 0.591, 0.811, 0.72, 0.85, 0.794],
        "XGBoost Classifier with depth 6": [0.62, 0.622, 0.585, 0.574, 0.798, 0.766, 0.868, 0.81]
    }
    st.table(version_3_data)

    st.markdown("### Best 3 Models")
    best_models_data = {
        "Metrics": [
            "Accuracy", "Recall", "Precision", "F1 Score", 
            "ROC AUC for class Negative", "ROC AUC for class Neutral", 
            "ROC AUC for class Positive", "Overall ROC-AUC"
        ],
        "SVM with linear kernel. C:0.1, 'auto' gamma": [0.643, 0.614, 0.636, 0.602, 0.832, 0.773, 0.882, 0.829],
        "MLP 'sgd' solver, layers(50,8), relu activation": [0.639, 0.614, 0.633, 0.606, 0.825, 0.782, 0.879, 0.829],
        "Naïve Bayes Multinomial": [0.633, 0.603, 0.624, 0.595, 0.822, 0.76, 0.858, 0.813]
    }
    st.table(best_models_data)

    st.markdown("## ROC Curve Analysis (TF-IDF Vectorization)")

    st.markdown(
        """
        The ROC curve compares the performance of the three best models using TF-IDF vectorization:

        - **Naive Bayes (Multinomial):** Represented by the solid blue line.
        - **SVM (Linear Kernel):** Represented by the dashed orange line.
        - **MLP (SGD Solver, ReLU Activation, 50x8 Layers):** Represented by the dashed green line with dots.

        ### Key Observations:
        - **SVM (Linear Kernel)** and **MLP (SGD Solver)** achieve the highest ROC-AUC scores (**0.829**), showing superior balance between sensitivity (True Positive Rate) and specificity (False Positive Rate).
        - **Naive Bayes (Multinomial)** performs slightly below the SVM and MLP models, with a competitive ROC-AUC score of **0.813**. This highlights its compatibility with TF-IDF for text classification.
        - The consistent performance across the three models reflects the strength of TF-IDF in capturing word importance and statistical patterns in sentiment analysis.

        The ROC curve highlights the ability of the models to distinguish between classes, with **SVM** and **MLP** emerging as the best performers.
        """
    )

    try:
        # Load and display the ROC curve image
        roc_image = Image.open("images/tfidf_roc_curve.jpeg")
        st.image(roc_image, caption="ROC Curve for the Best TF-IDF Models", use_column_width=True)
    except FileNotFoundError:
        st.error("ROC curve image not found. Please check the file path.")

    st.markdown("## Conclusions")

    st.markdown(
        """
        When using Word2Vec vectorization, the top three models are:
        1.⁠ ⁠XGBoost Classifier (Depth 6) - ROC-AUC: 0.732 (from Version 1)  
        2.⁠ ⁠MLP (Multi-Layer Perceptron) with 'lbfgs' solver, layers (64, 32, 16), ReLU activation - ROC-AUC: 0.716 (from Version 1)  
        3.⁠ ⁠SVM (Support Vector Machine) with linear kernel, C: 0.1, 'auto' gamma - ROC-AUC: 0.710 (from Version 2)  
        
        In contrast, when using TF-IDF vectorization, the top three models are:  
        1. MLP (Multi-Layer Perceptron) 
        2. SVM (Support Vector Machine)  
        3. Multinomial Naive Bayes  
        
        In terms of text classification for sentiment analysis, TF-IDF vectorization outperforms Word2Vec, especially when using SVM and MLP.  
        The improved performance is due to the fact that TF-IDF captures word importance based on term frequency and document frequency, which helps these models better identify and classify sentiment. SVM and MLP benefit particularly from this feature, as their architectures are better suited to differentiate sentiment in high-dimensional spaces where TF-IDF excels.  
        
        On the other hand, Word2Vec provides more nuanced semantic embeddings, which, while effective in some contexts, do not perform as well for sentiment classification tasks compared to TF-IDF, which is more directly aligned with capturing sentiment-based patterns in text.
        """
    )