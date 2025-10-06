import os
import argparse
import urllib.request
import zipfile
import tarfile
import shutil

def create_project_structure():
    """Create the project directory structure"""
    directories = [
        "data",
        "models",
        "samples"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_haar_cascade():
    """Download the Haar Cascade face detector model"""
    haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    haar_path = "models/haarcascade_frontalface_default.xml"
    
    print(f"Downloading Haar Cascade face detector model...")
    urllib.request.urlretrieve(haar_url, haar_path)
    print(f"Downloaded Haar Cascade face detector to {haar_path}")

def download_lfw_sample(sample_size=10):
    """Download a small subset of the LFW dataset for testing"""
    # URL for the LFW dataset (aligned version)
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
    temp_dir = "temp_lfw"
    lfw_tar = "temp_lfw/lfw-deepfunneled.tgz"
    extracted_dir = "temp_lfw/lfw-deepfunneled"
    
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download LFW dataset
    print(f"Downloading LFW dataset sample (this might take a while)...")
    urllib.request.urlretrieve(lfw_url, lfw_tar)
    
    # Extract the archive
    print("Extracting dataset...")
    with tarfile.open(lfw_tar, "r:gz") as tar:
        tar.extractall(path=temp_dir)
    
    # Copy a sample of the dataset
    print(f"Creating sample dataset with {sample_size} people...")
    sample_count = 0
    
    for person_name in os.listdir(extracted_dir):
        person_dir = os.path.join(extracted_dir, person_name)
        if os.path.isdir(person_dir) and sample_count < sample_size:
            # Check if there are at least 2 images
            images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= 2:
                # Create person directory in samples
                os.makedirs(f"samples/{person_name}", exist_ok=True)
                
                # Copy up to 5 images for this person
                for i, img_file in enumerate(images[:5]):
                    src = os.path.join(person_dir, img_file)
                    dst = os.path.join(f"samples/{person_name}", img_file)
                    shutil.copy(src, dst)
                
                sample_count += 1
                print(f"Added sample images for {person_name}")
    
    # Clean up
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    print(f"Sample dataset created with {sample_count} people.")

def create_empty_database():
    """Create an empty face database file"""
    import numpy as np
    
    # Create an empty dictionary and save it
    empty_db = {}
    np.save("data/face_database.npy", empty_db)
    print("Created empty face database file.")

def main():
    parser = argparse.ArgumentParser(description="Setup script for Face Recognition project")
    parser.add_argument("--download-samples", action="store_true", help="Download sample images from LFW dataset")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of people to sample from LFW")
    
    args = parser.parse_args()
    
    print("Setting up Face Recognition project...")
    create_project_structure()
    download_haar_cascade()
    create_empty_database()
    
    if args.download_samples:
        download_lfw_sample(args.sample_size)
    
    print("\nSetup complete! You can now run the app with: streamlit run app.py")

if __name__ == "__main__":
    main()