import cv2
import numpy as np
import os

# Global constants (same as used in the model)
IMG_SIZE = 96

def preprocess_image(image):
    """
    Preprocess an image for the face recognition model.
    
    Args:
        image: Input image (RGB or BGR)
        
    Returns:
        processed_image: Preprocessed image
    """
    # Ensure image is in RGB format
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3 and not isinstance(image[0, 0, 0], np.uint8):
        # If the image is already normalized, use it as is
        pass
    elif image.shape[2] == 3:
        # Check if the image is BGR (OpenCV default) and convert to RGB if needed
        if np.array_equal(image[0:5, 0:5, 0], cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[0:5, 0:5, 2]):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to the required input size for the model
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values
    if image.max() > 1.0:
        image = image / 255.0
    
    return image

def get_project_directories():
    """
    Create and return project directories.
    
    Returns:
        directories: Dictionary of project directory paths
    """
    # Define directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    # Create directories if they don't exist
    for directory in [data_dir, models_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    directories = {
        "base": base_dir,
        "data": data_dir,
        "models": models_dir
    }
    
    return directories

def load_sample_images(sample_dir="samples", num_samples=5):
    """
    Load sample images for testing.
    
    Args:
        sample_dir: Directory containing sample images
        num_samples: Maximum number of samples to load
        
    Returns:
        samples: List of loaded sample images
    """
    if not os.path.exists(sample_dir):
        return []
    
    samples = []
    count = 0
    
    for filename in os.listdir(sample_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and count < num_samples:
            img_path = os.path.join(sample_dir, filename)
            img = cv2.imread(img_path)
            samples.append((filename, img))
            count += 1
    
    return samples

def draw_text_with_background(image, text, position, font_scale=0.6, thickness=1,
                             text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Draw text with background on an image.
    
    Args:
        image: Input image
        text: Text to display
        position: (x, y) position for the text
        font_scale: Font scale factor
        thickness: Line thickness
        text_color: Text color (B, G, R)
        bg_color: Background color (B, G, R)
        
    Returns:
        image: Image with text overlay
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Calculate background rectangle dimensions
    rect_x, rect_y = position
    rect_width, rect_height = text_size[0] + 10, text_size[1] + 10
    
    # Draw background rectangle
    cv2.rectangle(image, (rect_x, rect_y), 
                 (rect_x + rect_width, rect_y + rect_height), 
                 bg_color, -1)
    
    # Draw text
    text_x = rect_x + 5
    text_y = rect_y + text_size[1] + 5
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    return image