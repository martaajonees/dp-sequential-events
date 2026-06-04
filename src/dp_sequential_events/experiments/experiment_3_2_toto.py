import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

def evaluate_dataset(dataset):
    # Load data 
    df = pd.read_csv(dataset)
    df = df.sort_values(by=['CaseID', 'Timestamp'])

    # Group by CaseID and create sequences
    sequences = df.groupby('CaseID')['Activity'].apply(list).values

    X_sequences = []
    Y_etiquettes = []
    for seq in sequences:
        if len(seq) > 1:  # Ensure there are at least 2 activities to predict the next one
            for i in range(1, len(seq)):
                X_sequences.append(seq[:i])  # Use all but the last activity as input
                Y_etiquettes.append(seq[i]) # Use the next activity as the label
    
    # Change the labels to integers
    all_activities = df['Activity'].unique()
    encoder= LabelEncoder()
    encoder.fit(all_activities)

    X_numerical = [encoder.transform(seq).tolist() for seq in X_sequences]
    Y_numerical = encoder.transform(Y_etiquettes)

    # Padding sequences to ensure they have the same length
    max_length = max(len(seq) for seq in X_numerical)
    X_padded = pad_sequences(X_numerical, maxlen=max_length, padding='pre')

    # Divide the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_padded, Y_numerical, test_size=0.2, random_state=42
    )

    # Build and train the LSTM model
    vocab_size = len(all_activities) + 1  # +1 for padding
    num_classes = len(all_activities)

    model = Sequential()
    # Layer 1 - Embedding layer
    model.add(Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length))
    # Layer 2 - LSTM layer
    model.add(LSTM(32))
    # Layer 3 - Final Dense layer with softmax activation for multi-class classification
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train the model
    print("\nTraining the LSTM model...")
    model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0)

    # Evaluate the model
    print("\nEvaluating the model...")
    _, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    dataset = input("\nEnter the dataset path: ")
    evaluate_dataset(dataset)