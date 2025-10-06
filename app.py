import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
import io

# Import our custom modules
from face_detection import detect_faces, extract_face
from face_recognition import load_feature_network, get_face_embedding, recognize_face
from database import load_face_database, add_face_to_database, create_person_folder
from utils import preprocess_image

# Set page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'database' not in st.session_state:
    st.session_state.database = None
if 'feature_network' not in st.session_state:
    st.session_state.feature_network = None
if 'detected_faces' not in st.session_state:
    st.session_state.detected_faces = []
if 'selected_face_idx' not in st.session_state:
    st.session_state.selected_face_idx = 0
if 'recognition_results' not in st.session_state:
    st.session_state.recognition_results = []
if 'add_person_mode' not in st.session_state:
    st.session_state.add_person_mode = False
if 'new_person_name' not in st.session_state:
    st.session_state.new_person_name = ""

# Function to load the model
@st.cache_resource
def load_model():
    return load_feature_network('models/feature_network.h5')

# Load the model and database
try:
    if st.session_state.feature_network is None:
        st.session_state.feature_network = load_model()
        st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    if st.session_state.database is None:
        st.session_state.database = load_face_database('data/face_database.npy')
        st.success(f"Database loaded with {len(st.session_state.database)} people")
except Exception as e:
    st.warning(f"Face database not found or couldn't be loaded: {e}. Starting with empty database.")
    st.session_state.database = {}

# Main app interface
st.title("Face Recognition with Siamese Neural Network")

# Sidebar
st.sidebar.header("Controls")

# Add new person mode toggle
add_person = st.sidebar.checkbox("Add New Person Mode", value=False)
st.session_state.add_person_mode = add_person

if st.session_state.add_person_mode:
    st.sidebar.subheader("Add New Person")
    st.session_state.new_person_name = st.sidebar.text_input("Enter person's name")
    st.sidebar.info("Upload an image with the person's face, then click 'Add to Database'")

# Recognition threshold setting
threshold = st.sidebar.slider(
    "Recognition Threshold",
    min_value=0.5,
    max_value=0.95,
    value=0.7,
    step=0.05,
    help="Higher values require stronger matches"
)

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
        
        # Detect faces in the image
        st.write("Detecting faces...")
        faces, face_locations = detect_faces(image)
        st.session_state.detected_faces = [(face, loc) for face, loc in zip(faces, face_locations)]
        
        if len(faces) > 0:
            st.success(f"Detected {len(faces)} face(s)")
        else:
            st.warning("No faces detected")

with col2:
    st.subheader("Detected Faces")
    
    if st.session_state.detected_faces:
        # Create a slider to select a face if there are multiple
        if len(st.session_state.detected_faces) > 1:
            face_idx = st.slider("Select Face", 0, len(st.session_state.detected_faces) - 1, 0)
            st.session_state.selected_face_idx = face_idx
        else:
            st.session_state.selected_face_idx = 0
        
        # Display the selected face
        selected_face, face_location = st.session_state.detected_faces[st.session_state.selected_face_idx]
        st.image(selected_face, caption=f"Face {st.session_state.selected_face_idx + 1}", width=250)
        
        # Recognize the face or add to database
        if st.session_state.add_person_mode:
            if st.button("Add to Database") and st.session_state.new_person_name:
                try:
                    # Preprocess the face image
                    processed_face = preprocess_image(selected_face)
                    
                    # Create directory for the person if it doesn't exist
                    create_person_folder(st.session_state.new_person_name)
                    
                    # Add face to database
                    add_face_to_database(
                        st.session_state.feature_network,
                        processed_face,
                        st.session_state.new_person_name,
                        st.session_state.database
                    )
                    
                    # Save the face image
                    timestamp = int(time.time())
                    img_path = f"data/{st.session_state.new_person_name}/{timestamp}.jpg"
                    cv2.imwrite(img_path, cv2.cvtColor(selected_face, cv2.COLOR_RGB2BGR))
                    
                    st.success(f"Added {st.session_state.new_person_name} to database!")
                except Exception as e:
                    st.error(f"Error adding face to database: {e}")
        else:
            if st.button("Recognize Face"):
                # Preprocess the face image
                processed_face = preprocess_image(selected_face)
                
                # Recognize the face
                person_name, similarity, distance = recognize_face(
                    st.session_state.feature_network,
                    processed_face,
                    st.session_state.database,
                    threshold
                )
                
                # Display recognition result
                if person_name != "Unknown":
                    st.success(f"✓ Recognized as: {person_name}")
                    st.info(f"Similarity: {similarity:.4f} (Distance: {distance:.4f})")
                else:
                    st.warning("❌ Unknown person")
                    st.info(f"Best match similarity: {similarity:.4f} (Distance: {distance:.4f})")

# Database management section
st.header("Database Management")
with st.expander("View Database"):
    if st.session_state.database:
        for person, embeddings in st.session_state.database.items():
            st.write(f"{person}: {len(embeddings)} face(s)")
        
        if st.button("Export Database"):
            # Create a downloadable version
            np.save('data/face_database_export.npy', st.session_state.database)
            with open('data/face_database_export.npy', 'rb') as f:
                st.download_button(
                    label="Download Database",
                    data=f,
                    file_name="face_database.npy",
                    mime="application/octet-stream"
                )
    else:
        st.info("Database is empty. Add some faces first!")

# About section
with st.expander("About This App"):
    st.markdown("""
    ### Face Recognition with Siamese Neural Network
    
    This app uses a Siamese Neural Network to perform face recognition. The network was trained to distinguish
    between different faces by learning similarity metrics between face images.
    
    **Features:**
    - Face detection
    - Face recognition
    - Add new faces to the database
    - Adjust recognition threshold
    
    **How to use:**
    1. Upload an image containing faces
    2. Select a face if multiple are detected
    3. Click "Recognize Face" to identify the person
    4. Or toggle "Add New Person Mode" to add new faces to the database
    
    Developed using TensorFlow, OpenCV, and Streamlit.
    """)

# Footer
st.markdown("---")
st.markdown(
    "Face Recognition System using Siamese Neural Networks | "
    "Built with Streamlit, TensorFlow, and OpenCV"
)