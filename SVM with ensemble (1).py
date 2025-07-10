from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import io

def load_dataset():
    from google.colab import files
    uploaded = files.upload()
    file_name = next(iter(uploaded))
    return pd.read_excel(io.BytesIO(uploaded[file_name]))

def preprocess_and_train(data):
    # Feature and target variables
    X = data[['protocol', 'num_bits', 'noise_level']]
    y = data['attack_type']

    # Preprocessing pipeline
    ct = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(), ["protocol"]),
            ("scaler", StandardScaler(), ['num_bits', 'noise_level'])
        ],
        remainder='passthrough'
    )

    # Define models
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    svm = SVC(kernel='rbf', C=1, probability=True, random_state=42)

    # Integrate all models into the ensemble
    estimators = [
        ('rf', rf),
        ('gb', gb),
        ('svm', svm)
    ]

    # Define ensemble model with VotingClassifier
    ensemble_model = StackingClassifier(
        estimators=estimators,
        final_estimator=VotingClassifier(estimators=estimators, voting='soft')
    )

    # Build full pipeline
    pipeline = ImbPipeline([
        ('preprocess', ct),
        ('sampling', SMOTE(random_state=42)),
        ('classifier', ensemble_model)
    ])

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    return pipeline, X_test, y_test, data

def predict_attack_success(model, original_data):
    try:
        # Get user input
        protocol = int(input("Enter protocol (1 for BB84 or 2 for B92): "))
        num_bits = int(input("Enter number of bits [64, 128, 256, 512, 1024, 2048, 4096]: "))
        noise_level = float(input("Enter noise level (%): "))

        # Create new data for prediction
        new_data = pd.DataFrame({
            'protocol': [protocol],
            'num_bits': [num_bits],
            'noise_level': [noise_level]
        })

        # Predict attack type
        predicted_attack = model.predict(new_data)[0]
        print("Predicted attack:", predicted_attack)

        # Fetch Eve's success rate from original data
        eve_success_rate = original_data.loc[original_data['attack_type'] == predicted_attack, 'eve_success_rate_estimate'].values[0]
        print("Eve's success rate for this attack:", eve_success_rate)

    except ValueError as e:
        print("Error:", e)

if _name_ == "_main_":
    original_data = load_dataset()

    model, X_test, y_test, data = preprocess_and_train(original_data.copy())

    predict_attack_success(model, original_data)