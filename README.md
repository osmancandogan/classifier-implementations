# MovieLens-100K Movie Recommendation Classifiers

This project implements various machine learning classifiers to predict user movie preferences using the MovieLens-100K dataset. The goal is to predict whether a user will like or dislike a movie based on available features.

## Dataset

The project uses the MovieLens-100K dataset which contains:
- 100,000 ratings (1-5) from 943 users on 1682 movies
- Each user has rated at least 20 movies
- Simple demographic info for the users

## Feature Engineering

The project employs sophisticated feature extraction techniques to convert movie data into machine-learning-ready features:

### Text Feature Extraction
- **TF-IDF (Term Frequency-Inverse Document Frequency)** is applied to movie titles
  - Converts text data into numerical features
  - Captures the importance of words in movie titles
  - Uses up to 500 features to represent title information
  - Removes common English stop words

### Genre Feature Processing
- **PCA (Principal Component Analysis)** is applied to movie genres
  - Reduces 18 binary genre features to 10 components
  - Preserves most important genre patterns
  - Helps reduce dimensionality while maintaining information
  - Combines related genres into meaningful components

## Models Implemented

The project implements five different classification models, each with its own strengths:

### 1. Logistic Regression
- Linear model for binary classification
- Fast and interpretable
- Works well with high-dimensional data
- Provides probability estimates
- Used as a baseline model

### 2. Decision Tree
- Non-linear model with tree structure
- Easily interpretable rules
- Can capture complex patterns
- Handles both numerical and categorical features
- Maximum depth of 10 to prevent overfitting

### 3. Random Forest
- Ensemble of 100 decision trees
- Reduces overfitting through averaging
- Better generalization than single trees
- Provides feature importance measures
- Good balance of accuracy and interpretability

### 4. XGBoost
- Advanced gradient boosting implementation
- High performance on structured data
- Handles missing values automatically
- Uses 100 trees for boosting
- Regularization to prevent overfitting

### 5. LightGBM
- Light Gradient Boosting Machine
- Faster training than traditional GBM
- Leaf-wise tree growth
- Good handling of categorical features
- Memory-efficient implementation

## License

This project is licensed under the MIT License - see the LICENSE file for details.