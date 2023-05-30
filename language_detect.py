import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load the data
data = pd.read_csv('language_data.csv')  # Assuming you have a CSV file with language data
X = data['text'].values
y = data['language'].values

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the input shape
input_shape = X_train.shape[1]

# Define the autoencoder architecture
input_layer = Input(shape=(input_shape,))
encoded = Dense(128, activation='relu')(input_layer)
decoded = Dense(input_shape, activation='sigmoid')(encoded)

# Define the encoder and decoder models
encoder = Model(input_layer, encoded)
autoencoder = Model(input_layer, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Extract the encoded features from the trained autoencoder
encoded_features = encoder.predict(X_train)

# Define the classification model on top of the encoded features
classification_input = Input(shape=(encoded_features.shape[1],))
classification_output = Dense(len(label_encoder.classes_), activation='softmax')(classification_input)
classification_model = Model(classification_input, classification_output)

# Compile the classification model
classification_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the classification model using the encoded features as input
classification_model.fit(encoded_features, y_train, epochs=50, batch_size=32, shuffle=True)

# Evaluate the model on the test set
encoded_test_features = encoder.predict(X_test)
loss, accuracy = classification_model.evaluate(encoded_test_features, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
