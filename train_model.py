import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class SimpleRegressor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        print(f"Data loaded from {file_path} with shape {df.shape}")
        return df

    def preprocess(self, df, target_column):
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        print(f"Preprocessing data: features shape {X.shape}, target shape {y.shape}")
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        print("Training model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Evaluation results: MSE = {mse}, R2 = {r2}")
        return mse, r2, predictions

    def plot_predictions(self, y_test, predictions):
        print("Plotting predictions...")
        plt.scatter(y_test, predictions, alpha=0.7)
        plt.xlabel('Actual Bandwidth')
        plt.ylabel('Predicted Bandwidth')
        plt.title('Actual vs Predicted Bandwidth')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal line
        plt.show()
    
    def predict(self, X):
        print("Making predictions...")
        return self.model.predict(X)

    def save_model(self, file_path):
        print(f"Saving model to {file_path}")
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, file_path):
        print(f"Loading model from {file_path}")
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
