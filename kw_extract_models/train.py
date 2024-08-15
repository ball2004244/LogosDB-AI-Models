'''
# First, generate train data using TF-IDF vectorizer
# train.csv format: RawText, Keywords
# e.g:
# RawText: "Wormholes, hypothetical topological features of spacetime, have long captivated the imagination of scientists and science fiction enthusiasts alike. These theoretical tunnels through space and time offer the tantalizing possibility of shortcuts across the universe, potentially allowing for faster-than-light travel. First conceptualized in 1935 by Einstein and Rosen, wormholes emerge from solutions to the equations of general relativity. While mathematically possible, the existence of traversable wormholes faces significant challenges. They would require exotic matter with negative energy density to remain open and stable, a concept that pushes the boundaries of known physics. If they exist, wormholes could connect distant regions of space-time, even linking different universes or timelines. This property has led to speculation about their potential for time travel, though the paradoxes this might create remain unresolved. Despite their theoretical intrigue, no observational evidence for wormholes has been found. Current research focuses on refining mathematical models and exploring potential detection methods. As our understanding of quantum gravity and the nature of spacetime evolves, wormholes continue to serve as a fascinating intersection of theoretical physics, cosmology, and our quest to unravel the universe's deepest mysteries.",
# Keywords: "wormholes, spacetime, theoretical physics, general relativity

# Then, train the model using the generated data
# The model must be able to handle multiple inputs and give multiple outputs instead of single input and single output of original TF-IDF model

# Finally, test the model using test data
# test.csv format: RawText, Keywords
'''

import time
import joblib
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

def preprocess_data(df, vectorizer=None, mlb=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['RawText'])
    else:
        X = vectorizer.transform(df['RawText'])

    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True)
        y = mlb.fit_transform(df['Keywords'].apply(
            lambda x: x.split(', ') if isinstance(x, str) else []))
    else:
        y = mlb.transform(df['Keywords'].apply(
            lambda x: x.split(', ') if isinstance(x, str) else []))

    return X, y, vectorizer, mlb

def build_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y, input_dim, output_dim):
    model = build_model(input_dim, output_dim)
    X_sparse = tf.sparse.SparseTensor(
        indices=np.array([X.row, X.col]).T,
        values=X.data,
        dense_shape=X.shape
    )
    y_sparse = tf.sparse.SparseTensor(
        indices=np.array([y.row, y.col]).T,
        values=y.data,
        dense_shape=y.shape
    )
    model.fit(X_sparse, y_sparse, epochs=10, batch_size=32, validation_split=0.1)
    return model

def evaluate_model(model, X, y):
    X_sparse = tf.sparse.SparseTensor(
        indices=np.array([X.row, X.col]).T,
        values=X.data,
        dense_shape=X.shape
    )
    y_sparse = tf.sparse.SparseTensor(
        indices=np.array([y.row, y.col]).T,
        values=y.data,
        dense_shape=y.shape
    )
    loss, accuracy = model.evaluate(X_sparse, y_sparse)
    return accuracy

def start_training():
    train_file = 'train.csv'
    valid_file = 'valid.csv'
    test_file = 'test.csv'
    model_file = 'model.h5'
    start = time.perf_counter()
    print('Reading training data...')

    df = pd.read_csv(train_file)
    print(f'Total number of samples: {len(df)}')

    print('Preprocessing data...')
    X, y, vectorizer, mlb = preprocess_data(df)

    print('Training model...')
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = train_model(X, y, input_dim, output_dim)

    print('Saving final model...')
    model.save(model_file)
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(mlb, 'mlb.pkl')
    print(f'Final model saved to {model_file}')

    print('Validating and testing the model...')
    valid_data = pd.read_csv(valid_file)
    test_data = pd.read_csv(test_file)

    X_valid, y_valid, _, _ = preprocess_data(valid_data, vectorizer, mlb)
    X_test, y_test, _, _ = preprocess_data(test_data, vectorizer, mlb)

    valid_accuracy = evaluate_model(model, X_valid, y_valid)
    test_accuracy = evaluate_model(model, X_test, y_test)

    print(f'Validation accuracy: {valid_accuracy:.2f}')
    print(f'Test accuracy: {test_accuracy:.2f}')
    print(f'Training completed!, Time taken: {time.perf_counter() - start:.2f} seconds')

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU devices found. Training will proceed on CPU.")
    start_training()