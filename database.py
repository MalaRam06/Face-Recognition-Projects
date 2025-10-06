import os
import numpy as np
from face_recognition import get_face_embedding

def create_person_folder(person_name):
    """
    Create a folder for storing a person's face images.
    
    Args:
        person_name: Name of the person
    """
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create person directory if it doesn't exist
    person_dir = os.path.join(data_dir, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    return person_dir

def load_face_database(database_path):
    """
    Load the face embedding database from disk.
    
    Args:
        database_path: Path to the saved database file
        
    Returns:
        database: Dictionary of face embeddings by person name
    """
    try:
        database = np.load(database_path, allow_pickle=True).item()
        print(f"Database loaded with {len(database)} people")
        return database
    except Exception as e:
        print(f"Failed to load database: {e}")
        return {}

def save_face_database(database, database_path="data/face_database.npy"):
    """
    Save the face embedding database to disk.
    
    Args:
        database: Dictionary of face embeddings by person name
        database_path: Path where to save the database
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    
    # Save the database
    np.save(database_path, database)
    print(f"Database saved with {len(database)} people")

def add_face_to_database(feature_network, face_img, person_name, database):
    """
    Add a new face embedding to the database.
    
    Args:
        feature_network: Loaded feature extraction model
        face_img: Preprocessed face image
        person_name: Name of the person
        database: Dictionary of face embeddings by person name
        
    Returns:
        database: Updated database
    """
    # Get embedding for the face
    embedding = get_face_embedding(feature_network, face_img)
    
    # Add to database
    if person_name not in database:
        database[person_name] = []
    
    database[person_name].append(embedding)
    
    # Save the updated database
    save_face_database(database)
    
    return database

def remove_person_from_database(person_name, database):
    """
    Remove a person from the face database.
    
    Args:
        person_name: Name of the person to remove
        database: Dictionary of face embeddings by person name
        
    Returns:
        database: Updated database
    """
    if person_name in database:
        del database[person_name]
        save_face_database(database)
        print(f"Removed {person_name} from database")
    else:
        print(f"{person_name} not found in database")
    
    return database

def get_database_stats(database):
    """
    Get statistics about the face database.
    
    Args:
        database: Dictionary of face embeddings by person name
        
    Returns:
        stats: Dictionary of database statistics
    """
    stats = {
        "total_people": len(database),
        "total_embeddings": sum(len(embeddings) for embeddings in database.values()),
        "people": {name: len(embeddings) for name, embeddings in database.items()}
    }
    
    return stats