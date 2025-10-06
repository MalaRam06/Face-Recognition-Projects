import os
import argparse
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Global constants
IMG_SIZE = 96

def load_and_preprocess_image(filepath):
    """Load and preprocess an image file."""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not read image at {filepath}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
    img = img / 255.0  # Normalize pixel values
    return img

def load_dataset(data_path):
    """Load dataset from directory structure."""
    images = []
    labels = []
    label_to_id = {}
    current_id = 0
    
    # Check if the directory exists
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory {data_path} does not exist")
    
    # Iterate through each person's directory
    for person_name in os.listdir(data_path):
        person_dir = os.path.join(data_path, person_name)
        if os.path.isdir(person_dir):
            # Assign an ID to this person if not already assigned
            if person_name not in label_to_id:
                label_to_id[person_name] = current_id
                current_id += 1
            
            # Load each image for this person
            for img_file in os.listdir(person_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_file)
                    try:
                        img = load_and_preprocess_image(img_path)
                        images.append(img)
                        labels.append(label_to_id[person_name])
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels), label_to_id

def create_pairs(images, labels, num_pairs=10000):
    """Create positive and negative pairs for Siamese network training."""
    pairs = []
    labels_pairs = []
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Dictionary to store indices for each label
    label_indices = {}
    for label in unique_labels:
        label_indices[label] = np.where(labels == label)[0]
    
    # Generate positive and negative pairs
    n_positive = num_pairs // 2
    n_negative = num_pairs - n_positive
    
    # Generate positive pairs (same person)
    for i in range(n_positive):
        # Select a random label that has at least 2 samples
        while True:
            label = np.random.choice(unique_labels)
            if len(label_indices[label]) >= 2:
                break
        
        # Select two different images from the same class
        idx1, idx2 = np.random.choice(label_indices[label], 2, replace=False)
        pairs.append([images[idx1], images[idx2]])
        labels_pairs.append(1)  # 1 for positive pair (same person)
    
    # Generate negative pairs (different people)
    for i in range(n_negative):
        # Select two different labels
        label1, label2 = np.random.choice(unique_labels, 2, replace=False)
        
        # Select one image from each class
        idx1 = np.random.choice(label_indices[label1])
        idx2 = np.random.choice(label_indices[label2])
        
        pairs.append([images[idx1], images[idx2]])
        labels_pairs.append(0)  # 0 for negative pair (different people)
    
    # Convert to numpy arrays and shuffle
    pairs = np.array(pairs)
    labels_pairs = np.array(labels_pairs)
    
    # Shuffle the pairs and labels together
    indices = np.random.permutation(len(pairs))
    return pairs[indices], labels_pairs[indices]

def create_base_network(input_shape):
    """Base network to be shared (CNN)."""
    model = Sequential([
        Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4)),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)),
        BatchNormalization(),
        
        Flatten(),
        
        Dense(512, activation='relu', kernel_regularizer=l2(1e-3)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=l2(1e-3))
    ])
    
    return model

def euclidean_distance(vectors):
    """Calculate the Euclidean distance between two vectors."""
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    """Output shape for Euclidean distance layer."""
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    """Contrastive loss function.
    Draws similar inputs closer and pushes dissimilar inputs apart.
    """
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_siamese_model(input_shape):
    """Create a Siamese network."""
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Create the base network (shared weights)
    base_network = create_base_network(input_shape)
    
    # Get the feature vectors for both images
    left_features = base_network(left_input)
    right_features = base_network(right_input)
    
    # Calculate the Euclidean distance between the feature vectors
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([left_features, right_features])
    
    # Define the Siamese network
    siamese_net = Model(inputs=[left_input, right_input], outputs=distance)
    
    # Compile the model
    optimizer = Adam(learning_rate=0.0001)
    siamese_net.compile(loss=contrastive_loss, optimizer=optimizer)
    
    return siamese_net, base_network

def compute_accuracy(predictions, labels):
    """Compute classification accuracy with a fixed threshold on distances."""
    threshold = 0.5
    return np.mean(np.equal(labels, predictions.reshape(-1) < threshold))

def train_model(data_path, epochs=20, batch_size=64, num_pairs=10000):
    """Train the Siamese network model."""
    # Load the dataset
    print(f"Loading dataset from {data_path}...")
    images, labels, label_to_id = load_dataset(data_path)
    print(f"Loaded {len(images)} images from {len(label_to_id)} people")
    
    # Create pairs for training
    print(f"Creating {num_pairs} pairs for training...")
    pairs, pair_labels = create_pairs(images, labels, num_pairs)
    print(f"Created {len(pairs)} pairs with {np.sum(pair_labels)} positive pairs")
    
    # Create and compile the model
    input_shape = (IMG_SIZE, IMG_SIZE, 3)  # RGB images
    print("Creating Siamese model...")
    siamese_model, feature_network = create_siamese_model(input_shape)
    siamese_model.summary()
    
    # Split the data into training and validation sets
    pairs_train, pairs_val, labels_train, labels_val = train_test_split(
        pairs, pair_labels, test_size=0.2, random_state=42
    )
    
    # Prepare the data for the model
    X_train_left = pairs_train[:, 0]
    X_train_right = pairs_train[:, 1]
    X_val_left = pairs_val[:, 0]
    X_val_right = pairs_val[:, 1]
    
    # Train the model
    print(f"Training model for {epochs} epochs...")
    history = siamese_model.fit(
        [X_train_left, X_train_right],
        labels_train,
        validation_data=([X_val_left, X_val_right], labels_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    
    # Evaluate the model
    val_predictions = siamese_model.predict([X_val_left, X_val_right])
    val_accuracy = compute_accuracy(val_predictions, labels_val)
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('models/training_history.png')
    
    # Save the models
    os.makedirs('models', exist_ok=True)
    siamese_model.save('models/siamese_model.h5')
    feature_network.save('models/feature_network.h5')
    print("Models saved to 'models/' directory")
    
    return siamese_model, feature_network, history

def main():
    parser = argparse.ArgumentParser(description="Train a Siamese network for face recognition")
    parser.add_argument("--data", type=str, default="samples", help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--pairs", type=int, default=10000, help="Number of pairs to generate for training")
    
    args = parser.parse_args()
    
    train_model(args.data, args.epochs, args.batch_size, args.pairs)

if __name__ == "__main__":
    main()