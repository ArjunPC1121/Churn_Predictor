
#optimised xgb
def train_xgboost():
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from xgboost import XGBClassifier
    
    # Load the dataset
    df = pd.read_csv("Modified_Churn_Modelling.csv")
    
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
    
    # Train the XGBoost model with updated parameters
    xgb = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.2, 
        random_state=42, 
        colsample_bytree=1.0, 
        subsample=1.0
    )
    
    print("Training XGBoost model...")
    xgb.fit(x_train, y_train)
    print("Model trained.")
    
    #Save the model and scaler
    try:
        joblib.dump(xgb, 'xgboost_model.pkl')
        joblib.dump(scaler, 'scaler_xgb.pkl')
        print("Model and scaler saved successfully.")
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")
    
    # Evaluate the model
    probs = xgb.predict_proba(x_test)[:, 1]  # Predicted probabilities for the positive class
    predictions = xgb.predict(x_test)       # Predicted labels
    print("\nModel Evaluation:")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.2f}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(f"F1-Score: {f1_score(y_test, predictions):.2f}")
    
    # Display feature importance (optional, you can uncomment this if needed)
    # print("\nFeature Importance:")
    # for feature, importance in zip(selected_features, xgb.feature_importances_):
    #     print(f"{feature}: {importance:.4f}")

train_xgboost()
