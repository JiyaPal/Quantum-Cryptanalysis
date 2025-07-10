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
from imblearn.pipeline import Pipeline as ImbPipeline
from tensorflow.keras.optimizers import Adam


def load_dataset():
    from google.colab import files
    uploaded = files.upload()
    file_name = next(iter(uploaded))
    return pd.read_excel(io.BytesIO(uploaded[file_name]))


def create_ann_model(input_dim, num_classes):
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
    # Filter out class 0
    data = data[data['attack_type'] != 0]

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

    # ANN model input dimension and number of classes
    input_dim = X_resampled.shape[1]
    num_classes = y_resampled_categorical.shape[1]

    # Create ANN model
    model = create_ann_model(input_dim, num_classes)

    # Train/Test split using StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled_categorical[train_index], y_resampled_categorical[test_index]

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_classes = y_pred.argmax(axis=1)
        y_test_classes = y_test.argmax(axis=1)

        print("Classification Report:")
        print(classification_report(y_test_classes, y_pred_classes))

    return model, X_test, y_test, data, ct


def predict_attack_success(model, original_data, column_transformer):
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
        predicted_attack = y_pred.argmax(axis=1)[0]
        print("Predicted attack:", predicted_attack)

        # Fetch Eve's success rate from original data
        eve_success_rate = original_data.loc[original_data['attack_type'] == predicted_attack, 'eve_success_rate_estimate'].values[0]
        print("Eve's success rate for this attack:", eve_success_rate)

    except ValueError as e:
        print("Error:", e)


if _name_ == "_main_":
    original_data = load_dataset()

    model, X_test, y_test, data, ct = preprocess_and_train(original_data.copy())

    predict_attack_success(model, original_data, ct)