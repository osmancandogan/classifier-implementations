import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils.movielens_get_df import MovieLensDataset

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4)
    }

def main():
    print("Loading MovieLens dataset...")
    dataset = MovieLensDataset(ratings_file='ml-100k/u.data', movies_file='ml-100k/u.item')
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100),
        'LightGBM': LGBMClassifier(random_state=42, n_estimators=100)
    }
    
    results = []
    for name, model in models.items():
        result = evaluate_model(
            model, 
            dataset.X_train, 
            dataset.X_test, 
            dataset.y_train, 
            dataset.y_test,
            name
        )
        results.append(result)
    
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")

if __name__ == "__main__":
    main() 