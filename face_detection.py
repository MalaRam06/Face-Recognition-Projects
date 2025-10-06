import cv2
import numpy as np

# Load the pre-trained face detection model
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading face cascade: {e}")
    # Fallback path
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def detect_faces(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Detect faces in an image and return the cropped faces and their locations.
    
    Args:
        image: Input image (BGR)
        scale_factor: Parameter specifying how much the image size is reduced at each image scale
        min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
        min_size: Minimum possible object size
        
    Returns:
        faces: List of cropped face images (RGB)
        locations: List of face locations as (x, y, w, h)
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_locations = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    
    # Extract each face
    faces = []
    for (x, y, w, h) in face_locations:
        face_image = image[y:y+h, x:x+w]
        # Convert to RGB for display
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        faces.append(face_rgb)
    
    return faces, face_locations

def extract_face(image, face_location, margin=0):
    """
    Extract a face from an image based on location coordinates.
    
    Args:
        image: Input image
        face_location: (x, y, w, h) coordinates
        margin: Optional margin to add around the face
        
    Returns:
        face_image: Cropped face image
    """
    x, y, w, h = face_location
    
    # Add margin if specified
    if margin > 0:
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
    
    # Extract the face
    face_image = image[y:y+h, x:x+w]
    
    return face_image

def draw_face_boxes(image, face_locations, labels=None, colors=None):
    """
    Draw boxes around detected faces with optional labels.
    
    Args:
        image: Input image
        face_locations: List of (x, y, w, h) face coordinates
        labels: Optional list of labels for each face
        colors: Optional list of colors for each face box
        
    Returns:
        image_with_boxes: Image with face boxes drawn
    """
    image_with_boxes = image.copy()
    
    for i, (x, y, w, h) in enumerate(face_locations):
        # Default color is green
        color = (0, 255, 0)
        if colors and i < len(colors):
            color = colors[i]
        
        # Draw the rectangle around the face
        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), color, 2)
        
        # Add label if provided
        if labels and i < len(labels):
            label = labels[i]
            cv2.putText(image_with_boxes, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image_with_boxes