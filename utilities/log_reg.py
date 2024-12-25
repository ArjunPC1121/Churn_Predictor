import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
def calculateChurn(age, bal, binary_active, prod, gender_Male):
    #print(f"Inputs - Age: {age}, Balance: {bal}, Active: {binary_active}, Products: {prod}, Gender_Male: {gender_Male}")
    df = pd.read_csv("Modified_Churn_Modelling.csv")

    # x includes all columns except 'Exited'
    x = df.drop(columns=['Exited'])
    y = df['Exited']

    # One-hot Encoding
    x = pd.get_dummies(x, drop_first=True)

    # Drop location-based features before feature selection
    x = x.drop(columns=[col for col in x.columns if 'Geography_' in col])

    # Selecting 5 best features
    selector = SelectKBest(score_func=f_classif, k=5)
    x_new = selector.fit_transform(x, y)
    selected_features = x.columns[selector.get_support()]

    print("Selected features after excluding location: ")
    for feature in selected_features:
        print(feature)
    # Features selected by SelectKBest: Example - Age, Balance, NumOfProds,IsActiveMember,Gender.

    # Scaling the data
    scaler = StandardScaler()
    x_new = scaler.fit_transform(x_new)

    # Splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.33, random_state=19)

    # Model training
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_train, y_train)

    # Save the model and scaler for future use
    joblib.dump(lr, 'logistic_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Model evaluation
    probs = lr.predict_proba(x_test)[:, 1]
    print("Model Evaluation:")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.2f}")
    print(f"Accuracy: {accuracy_score(y_test, lr.predict(x_test)):.2f}")
    print(f"F1-Score: {f1_score(y_test, lr.predict(x_test)):.2f}")

    '''# Display predicted probabilities for the first 10 test cases
    print("Predicted Probabilities for the first 10 customers in the test set:")
    for i, prob in enumerate(probs[:10]):
        print(f"Customer {i+1}: {prob:.2f}")'''

    # Accepting user input for prediction
    '''print("Enter the following details to predict churn probability:")
    age = int(input("Enter Age:"))
    bal = float(input("Enter Balance:"))
    active = input("Is the customer active? (yes/no):").strip().lower()
    prod = int(input("Enter Number of products:"))
    gender = input("Enter Gender (male/female):").strip().lower()

     Converting input to required form
    binary_active = 1 if active == 'yes' else 0
    gender_Male = 1 if gender == 'male' else 0

    # Creating input vector based on selected features
    l = [[age, bal, binary_active, prod, gender_Male]]

    # Scaling the input
    l_scaled = scaler.transform(l)

    # Predicting probability
    res = lr.predict_proba(l_scaled)[:, 0]

    # Displaying the prediction
    #print(f"\nThe probability of churn for the given customer is: {res[0] * 100:.2f}%")
    return res'''
    
calculateChurn(25,765898,1,2,1)
