import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Global constants
IMG_SIZE = 100  # Same as used in training

def load_feature_network(model_path):
    """
    Load the pre-trained feature extraction network.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        model: Loaded Keras model
    """
    try:
        # Load the model
        model = load_model(model_path, compile=False)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {e}")

def get_face_embedding(feature_network, face_img):
    """
    Convert a face image to an embedding vector using the trained network.
    
    Args:
        feature_network: Loaded feature extraction model
        face_img: Preprocessed face image
        
    Returns:
        embedding: Face embedding vector
    """
    # Ensure image is properly preprocessed
    if face_img.shape != (IMG_SIZE, IMG_SIZE, 3):
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        
    if face_img.max() > 1.0:
        face_img = face_img / 255.0
        
    # Add batch dimension
    face_img = np.expand_dims(face_img, axis=0)
    
    # Get embedding vector
    embedding = feature_network.predict(face_img, verbose=0)
    
    return embedding[0]  # Return as 1D array
def calculate_similarity(embedding1, embedding2, method='cosine'):
    """
    Calculate similarity between two face embeddings.
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
        method: 'cosine' or 'euclidean'
        
    Returns:
        similarity: Similarity score (0-1, higher means more similar)
        distance: Original distance measure
    """
    if method == 'cosine':
        # Normalize embeddings to unit length
        norm_emb1 = embedding1 / np.linalg.norm(embedding1)
        norm_emb2 = embedding2 / np.linalg.norm(embedding2)
        
        # Cosine similarity (dot product of normalized vectors)
        similarity = np.dot(norm_emb1, norm_emb2)
        distance = 1 - similarity
    else:
        # Euclidean distance (original method)
        distance = np.linalg.norm(embedding1 - embedding2)
        similarity = 1 / (1 + distance)
        
    return similarity, distance

def recognize_face(feature_network, face_img, database, threshold=0.85, method='cosine'):
    """
    Recognize a face by comparing its embedding with the database.
    
    Args:
        feature_network: Loaded feature extraction model
        face_img: Preprocessed face image
        database: Dictionary of face embeddings by person name
        threshold: Similarity threshold for recognition
        method: Similarity method ('cosine' or 'euclidean')
        
    Returns:
        person_name: Recognized person's name or "Unknown"
        similarity: Best match similarity score
        distance: Best match distance value
    """
    # Get embedding for the input face
    embedding = get_face_embedding(feature_network, face_img)
    
    if not database:
        return "Unknown", 0, float('inf')
    
    # Store all similarity scores for debugging
    all_scores = {}
    
    best_match = None
    best_similarity = -1
    best_distance = float('inf')
    
    # Compare with each person in the database
    for person_name, embeddings in database.items():
        person_similarities = []
        
        # Compare with all embeddings for this person
        for stored_embedding in embeddings:
            similarity, distance = calculate_similarity(embedding, stored_embedding, method=method)
            person_similarities.append(similarity)
        
        # Use the average similarity for this person (more robust)
        avg_similarity = np.mean(person_similarities)
        all_scores[person_name] = avg_similarity
        
        if avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_match = person_name
    
    # Print all scores for debugging
    print(f"All similarity scores: {all_scores}")
    
    # Check if the best match exceeds the threshold
    if best_similarity >= threshold:
        return best_match, best_similarity, best_distance
    else:
        return "Unknown", best_similarity, best_distance