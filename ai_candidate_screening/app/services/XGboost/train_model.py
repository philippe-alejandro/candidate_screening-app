import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import joblib
import os

# Command to run file: python3 ai_candidate_screening/app/services/XGboost/train_model.py 

def train_xgboost_model(csv_path=None, model_save_dir=None):
    try:
        print("Starting training process...")

        # Dynamically resolve the absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if csv_path is None:
            csv_path = os.path.join(script_dir, "../../data/refined_training_data.csv")
        if model_save_dir is None:
            model_save_dir = os.path.join(script_dir)

        print("Resolved CSV Path:", csv_path)
        print("Resolved Model Save Directory:", model_save_dir)

        # Ensure the save directory exists
        os.makedirs(model_save_dir, exist_ok=True)
        print("Ensured save directory exists.")

        # Load the training data
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

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

        # Convert data into DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Define parameters for the XGBoost model
        params = {
            "objective": "reg:squarederror",  # Regression problem
            "eval_metric": "rmse",           # Root Mean Squared Error
            "max_depth": 6,                  # Maximum depth of trees
            "eta": 0.1,                      # Learning rate
            "subsample": 0.8,                # Subsampling ratio
            "colsample_bytree": 0.8          # Fraction of features used per tree
        }

        print("Training model...")
        evals = [(dtrain, "train"), (dtest, "test")]
        model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)
        print("Model trained.")

        # Save the trained model
        model_path = os.path.join(model_save_dir, "model.xgb")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

        # Save preprocessors
        preprocessor_path = os.path.join(model_save_dir, "preprocessors.pkl")
        joblib.dump({"scaler": scaler, "encoder": encoder}, preprocessor_path)
        print(f"Preprocessors saved to {preprocessor_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train_xgboost_model()
