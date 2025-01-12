import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class MovieLensDataset:
    # singleton class
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MovieLensDataset, cls).__new__(cls)
        return cls._instance

    def __init__(self, ratings_file, movies_file, threshold=3.5):
        ratings = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )

        movies = pd.read_csv(
            movies_file,
            sep='|',
            names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
            usecols=['movie_id', 'title', 'release_date', 'Action', 'Adventure', 'Animation',
                     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western'],
            encoding='latin-1'
        )

        ratings['like'] = (ratings['rating'] >= threshold).astype(int)

        data = pd.merge(ratings, movies, on='movie_id')
        data['title'] = data['title'].fillna('')
        self.data = data
        self.X, self.y = self.feature_extraction()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split()

    def feature_extraction(self, max_features=500, n_components=10, random_state=42):
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        title_tfidf = tfidf_vectorizer.fit_transform(self.data['title'])

        genre_features = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                          'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                          'Thriller', 'War', 'Western']
        pca = PCA(n_components=n_components, random_state=random_state)
        genre_pca = pca.fit_transform(self.data[genre_features])

        combined_features = np.hstack([title_tfidf.toarray(), genre_pca])

        combined_features_df = pd.DataFrame(combined_features)
        X = combined_features_df
        y = self.data['like']

        return X, y

    def split(self, test_size=0.2, random_state=42):
        X, y = self.feature_extraction()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test
