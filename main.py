import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.variances = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.means[cls] = np.mean(X_cls, axis=0)
            self.variances[cls] = np.var(X_cls, axis=0) + 1e-6  # Add epsilon to avoid zero variance
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def _gaussian_pdf(self, x, mean, var):
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2.0 * var))
        return coeff * exponent

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                likelihood = np.sum(np.log(self._gaussian_pdf(x, self.means[cls], self.variances[cls])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

# Load the dataset
data = pd.read_csv('cardio_train.csv', sep=';')

# Preprocess the data
# Remove the 'id' column if it exists and separate features and target
if 'id' in data.columns:
    data = data.drop(columns=['id'])

X = data.drop(columns=['cardio']).values
y = data['cardio'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the custom Naive Bayes classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# Perform k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_index, val_index in kf.split(X_train):
    X_kf_train, X_kf_val = X_train[train_index], X_train[val_index]
    y_kf_train, y_kf_val = y_train[train_index], y_train[val_index]

    nb_classifier.fit(X_kf_train, y_kf_train)
    y_kf_pred = nb_classifier.predict(X_kf_val)
    accuracy = accuracy_score(y_kf_val, y_kf_pred)
    accuracies.append(accuracy)

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

# Test the model on the test set
y_test_pred = nb_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
print("Cross-Validation Results:")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print("\nTest Set Results:")
print(f"Test Set Accuracy: {test_accuracy:.4f}")