import pandas as pd
import io
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from tensorflow.keras.optimizers import Adam


def load_dataset():
    """Load dataset via file upload"""
    from google.colab import files
    uploaded = files.upload()
    file_name = next(iter(uploaded))
    return pd.read_excel(io.BytesIO(uploaded[file_name]))


def create_ann_model(input_dim, num_classes):
    """Create and compile the ANN model"""
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def preprocess_and_train(data):
    """Preprocess data and train the model using cross-validation"""
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

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(ct.fit_transform(X), y)

    # Convert target to categorical
    y_resampled_categorical = to_categorical(pd.factorize(y_resampled)[0])

    # Map the integer labels back to the original attack types
    attack_types_mapping = dict(enumerate(pd.unique(y_resampled)))

    # ANN model input dimension and number of classes
    input_dim = X_resampled.shape[1]
    num_classes = y_resampled_categorical.shape[1]

    # Create the ANN model
    model = create_ann_model(input_dim, num_classes)

    # Train/Test split using StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled_categorical[train_index], y_resampled_categorical[test_index]

        # Train the model (increase epochs to allow more learning time)
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_data=(X_test, y_test))

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_classes = y_pred.argmax(axis=1)
        y_test_classes = y_test.argmax(axis=1)

        print("Classification Report:")
        print(classification_report(y_test_classes, y_pred_classes, zero_division=1))

    return model, X_test, y_test, ct, attack_types_mapping, data


def predict_attack_success(model, original_data, column_transformer, attack_types_mapping):
    """Predict the attack type and success rate based on user input"""
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

        # Preprocess the new data
        new_data_transformed = column_transformer.transform(new_data)

        # Predict attack type
        y_pred = model.predict(new_data_transformed)
        predicted_attack_index = y_pred.argmax(axis=1)[0]

        # Map the predicted index back to the original attack type
        predicted_attack = attack_types_mapping[predicted_attack_index]
        print("Predicted attack:", predicted_attack)

        # Check if the predicted attack type exists in the original data
        matching_attack = original_data.loc[original_data['attack_type'] == predicted_attack]

        if not matching_attack.empty:
            # Fetch Eve's success rate from original data
            eve_success_rate = matching_attack['eve_success_rate_estimate'].values[0]
            print("Eve's success rate for this attack:", eve_success_rate)
        else:
            print(f"No matching attack type found for predicted attack {predicted_attack}.")

    except ValueError as e:
        print("Error:", e)


if __name__ == "__main__":
    # Load dataset
    original_data = load_dataset()

    # Preprocess data and train the model
    model, X_test, y_test, column_transformer, attack_types_mapping, data = preprocess_and_train(original_data.copy())

    # Predict attack success based on user input
    predict_attack_success(model, original_data, column_transformer, attack_types_mapping)
