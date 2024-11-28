import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# Command to run file: python ai_candidate_screening/app/services/neural_network/train_model.py 

def train_neural_network(csv_path, model_save_dir="app/services/neural_ network"):
    try:
        print("Starting training process...")
        os.makedirs(model_save_dir, exist_ok=True)
        print("Ensured save directory exists.")

        # Load the training data
        print(f"CSV Path: {csv_path}")
        df = pd.read_csv(csv_path)
        print("Data loaded successfully:", df.head())

        # Separate features and target
        print("Separating features and target...")
        X = df[["Experience_Score", "Skills_Score", "Education_Score"]]  # Use only available feature columns
        y = df["Score"]  # Target column

        # Preprocess features
        print("Preprocessing features...")
        encoder = LabelEncoder()
        for col in ["Skills_Score", "Education_Score"]:  # Encode only "Skills" and "Education"
            X[col] = encoder.fit_transform(X[col].astype(str))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Features standardized.")

        # Split data into training and testing sets
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Build a neural network model
        print("Building model...")
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))  # Regression output (Score)

        print("Compiling model...")
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        print("Training model...")
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
        print("Model trained.")

        # Save the trained model
        model_path = f"{model_save_dir}/model.h5"
        model.save(model_path)
        print(f"Model saved to {model_path}")

        # Save preprocessors
        preprocessor_path = f"{model_save_dir}/preprocessors.pkl"
        joblib.dump({"scaler": scaler, "encoder": encoder}, preprocessor_path)
        print(f"Preprocessors saved to {preprocessor_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    csv_path = "app/data/refined_training_data.csv"  # Update with actual CSV path
    train_neural_network(csv_path)