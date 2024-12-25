import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Load and train the Random Forest model
def train_random_forest():
    # Load the dataset
    df = pd.read_csv("Modified_Churn_Modelling.csv")  # Adjust the path as needed

    # Split data into features and target
    x = df.drop(columns=['Exited'])
    y = df['Exited']

    # One-hot encoding for categorical variables
    x = pd.get_dummies(x, drop_first=True)

    # Drop location-based features
    x = x.drop(columns=[col for col in x.columns if 'Geography_' in col])

    # Select specific features for training
    selected_features = ['Age', 'Balance', 'IsActiveMember', 'NumOfProducts', 'Gender_Male']
    x = x[selected_features]

    # Standardize the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state=19)

    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=700, max_leaf_nodes=16, random_state=42)
    print("Training Random Forest model...")
    rf.fit(x_train, y_train)
    print("Model trained.")

    # Save the model and scaler
    try:
        joblib.dump(rf, 'random_forest_model.pkl')
        joblib.dump(scaler, 'scaler_1.pkl')
        print("Model and scaler saved successfully.")
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")

    # Evaluate the model
    probs = rf.predict_proba(x_test)[:, 1]  # Predicted probabilities for the positive class
    predictions = rf.predict(x_test)       # Predicted labels

    print("\nModel Evaluation:")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.2f}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(f"F1-Score: {f1_score(y_test, predictions):.2f}")

    # Display feature importance
    print("\nFeature Importance:")
    for feature, importance in zip(selected_features, rf.feature_importances_):
        print(f"{feature}: {importance:.4f}")

# Main function to call train_random_forest()
if __name__ == "__main__":
    train_random_forest()