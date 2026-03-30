# Face Recognition System with Siamese Neural Networks

This project implements a face recognition system using Siamese Neural Networks and provides a Streamlit web interface for easy interaction.

## Features

- Face detection using OpenCV
- Face recognition using Siamese Neural Networks
- Add new faces to the database through the UI
- Adjust recognition threshold based on confidence needs
- Simple and intuitive web interface

## Project Structure

```
face-recognition-project/
├── app.py                      # Main Streamlit application
├── face_detection.py           # Face detection utilities
├── face_recognition.py         # Face recognition functions
├── database.py                 # Database management
├── utils.py                    # Utility functions
├── requirements.txt            # Project dependencies
├── models/                     # Trained models
│   └── feature_network.h5      # Trained feature extraction network
├── data/                       # Face database and images
│   └── face_database.npy       # Saved face embeddings
└── notebooks/                  # Jupyter notebooks
    └── Face_Recognition_Siamese_Network.ipynb  # Model training notebook
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Face-Recognition-Projects.git
cd Face-Recognition-Projects
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model using the provided notebook or download the pre-trained model.

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501).

3. Use the application:
   - Upload an image containing faces
   - The app will detect and highlight faces in the image
   - Select a face if multiple are detected
   - Click "Recognize Face" to identify the person
   - Use "Add New Person Mode" to add new faces to the database

## How the System Works

### Siamese Neural Network

The system uses a Siamese Neural Network architecture to learn face embeddings. This approach is particularly effective for face recognition tasks because:

1. It learns to differentiate between pairs of faces
2. It works well with limited training data per person
3. New people can be added to the system without retraining the model

### Face Recognition Process

1. Face Detection: OpenCV's Haar Cascade classifier is used to detect faces in images
2. Face Embedding: The detected face is passed through the trained neural network to generate a 128-dimensional embedding vector
3. Similarity Comparison: The embedding is compared to stored embeddings in the database using Euclidean distance
4. Recognition Decision: If the similarity exceeds the threshold, the face is recognized as the matching person

## Model Training

The Siamese network model was trained on face image pairs with a contrastive loss function. See the Jupyter notebook in the `notebooks` directory for the complete training process.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Streamlit
- NumPy
- scikit-learn

## License

MIT License
