import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils.movielens_get_df import MovieLensDataset

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model using single train-test split"""
    print(f"\nTraining {model_name} (Single Split)...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'Model': f"{model_name} (Single Split)",
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4)
    }

def evaluate_model_kfold(model, dataset, model_name, n_splits=5):
    """Evaluate model using k-fold cross-validation"""
    print(f"\nTraining {model_name} (K-Fold)...")
    splits = dataset.get_kfold_splits(n_splits=n_splits)
    fold_scores = []
    
    for i, (X_train, X_test, y_train, y_test) in enumerate(splits, 1):
        print(f"Training fold {i}/{n_splits}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        scores = {
            'Fold': i,
            'Model': f"{model_name} (Fold {i})",
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4)
        }
        fold_scores.append(scores)
    
    # Calculate average scores across folds
    avg_scores = {
        'Model': f"{model_name} (K-Fold Average)",
        'Accuracy': round(np.mean([s['Accuracy'] for s in fold_scores]), 4),
        'Precision': round(np.mean([s['Precision'] for s in fold_scores]), 4),
        'Recall': round(np.mean([s['Recall'] for s in fold_scores]), 4)
    }
    
    return fold_scores, avg_scores

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
    
    # Results storage
    single_split_results = []
    kfold_detailed_results = []
    kfold_avg_results = []
    
    # Evaluate each model with both methods
    for name, model in models.items():
        # Single train-test split evaluation
        single_result = evaluate_model(
            model, 
            dataset.X_train, 
            dataset.X_test, 
            dataset.y_train, 
            dataset.y_test,
            name
        )
        single_split_results.append(single_result)
        
        # K-fold cross-validation
        fold_results, avg_result = evaluate_model_kfold(model, dataset, name)
        kfold_detailed_results.extend(fold_results)
        kfold_avg_results.append(avg_result)
    
    # Create DataFrames for results
    single_split_df = pd.DataFrame(single_split_results)
    kfold_detailed_df = pd.DataFrame(kfold_detailed_results)
    kfold_avg_df = pd.DataFrame(kfold_avg_results)
    
    # Print results
    print("\n=== Single Split Results ===")
    print(single_split_df.to_string(index=False))
    
    print("\n=== K-Fold Cross-Validation Results (Detailed) ===")
    print(kfold_detailed_df.to_string(index=False))
    
    print("\n=== K-Fold Cross-Validation Results (Averaged) ===")
    print(kfold_avg_df.to_string(index=False))
    
    # Save results
    single_split_df.to_csv('single_split_results.csv', index=False)
    kfold_detailed_df.to_csv('kfold_detailed_results.csv', index=False)
    kfold_avg_df.to_csv('kfold_avg_results.csv', index=False)
    
    print("\nResults saved to CSV files:")
    print("- single_split_results.csv")
    print("- kfold_detailed_results.csv")
    print("- kfold_avg_results.csv")

if __name__ == "__main__":
    main() 