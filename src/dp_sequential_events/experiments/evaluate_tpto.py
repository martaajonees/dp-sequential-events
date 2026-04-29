import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

def extract_sequences(df):
    sequences = df.groupby('CaseID')['Activity'].apply(list).values
    X_seqs, Y_labels = [], []
    
    for seq in sequences:
        if len(seq) > 1:
            for i in range(1, len(seq)):
                X_seqs.append(seq[:i])
                Y_labels.append(seq[i])
                
    return X_seqs, Y_labels


def evaluate_dataset(dataset, private):
   # Load data 
    df_orig = pd.read_csv(dataset).sort_values(by=['CaseID', 'Timestamp'])
    df_priv = pd.read_csv(private).sort_values(by=['CaseID', 'Timestamp'])

    # Extract sequences and labels from both datasets
    X_orig_raw, Y_orig_raw = extract_sequences(df_orig)
    X_priv_raw, Y_priv_raw = extract_sequences(df_priv)
    
    # Change the labels to integers
    all_activities = df_orig['Activity'].unique()
    encoder = LabelEncoder()
    encoder.fit(all_activities)

    # Transform sequences and labels to numerical format
    X_orig_num = [encoder.transform(seq).tolist() for seq in X_orig_raw]
    Y_orig_num = encoder.transform(Y_orig_raw)

    X_priv_num = [encoder.transform(seq).tolist() for seq in X_priv_raw]
    Y_priv_num = encoder.transform(Y_priv_raw)

    # Padding sequences to ensure they have the same length
    max_length = max(len(seq) for seq in X_orig_num)

    X_orig_padded = pad_sequences(X_orig_num, maxlen=max_length, padding='pre')
    X_priv_padded = pad_sequences(X_priv_num, maxlen=max_length, padding='pre')

    # Divide the dataset into training and testing sets
    # Test set from the original dataset
    _, X_test, _, Y_test = train_test_split(
        X_orig_padded, Y_orig_num, test_size=0.2, random_state=42
    )
    # Training set from the private dataset
    X_train, _, Y_train, _ = train_test_split(
        X_priv_padded, Y_priv_num, test_size=0.2, random_state=42
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
    private = input("\nEnter the private dataset path: ")
    evaluate_dataset(dataset, private)