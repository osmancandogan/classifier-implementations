from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from movielens_get_df import MovieLensDataset

dataset = MovieLensDataset(ratings_file='ml-100k/u.data', movies_file='ml-100k/u.item')
rf_model = RandomForestClassifier(random_state=42, n_estimators=100,max_depth=10)  # Adjust max_depth for complexity
rf_model.fit(dataset.X_train, dataset.y_train)
y_pred = rf_model.predict(dataset.X_test)

print("Accuracy:", accuracy_score(dataset.y_test, y_pred))
print("Precision:", precision_score(dataset.y_test, y_pred))
print("Recall:", recall_score(dataset.y_test, y_pred))
print("\nClassification Report:")
print(classification_report(dataset.y_test, y_pred))
