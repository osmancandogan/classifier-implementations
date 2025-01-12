from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from movielens_get_df import MovieLensDataset

dataset = MovieLensDataset(ratings_file='ml-100k/u.data', movies_file='ml-100k/u.item')
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(dataset.X_train, dataset.y_train)
y_pred = dt_model.predict(dataset.X_test)

print("Accuracy:", accuracy_score(dataset.y_test, y_pred))
print("Precision:", precision_score(dataset.y_test, y_pred))
print("Recall:", recall_score(dataset.y_test, y_pred))
print("\nClassification Report:")
print(classification_report(dataset.y_test, y_pred))

"""
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=dataset.X.columns,
    class_names=['Dislike', 'Like'],
    filled=True,
    rounded=True
)
plt.savefig("decision_tree_high_res.png", dpi=400, bbox_inches='tight')
plt.show()
"""
