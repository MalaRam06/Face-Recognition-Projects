#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys

def check_requirements():
    """Check if all requirements are installed."""
    print("Checking requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All requirements are installed.")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install requirements.")
        return False

def setup_project():
    """Set up the project structure and download necessary files."""
    print("Setting up project...")
    try:
        subprocess.check_call([sys.executable, "setup.py"])
        print("Project setup completed.")
        return True
    except subprocess.CalledProcessError:
        print("Failed to set up project.")
        return False

def train_model(data_path, epochs):
    """Train the face recognition model."""
    print(f"Training model with data from {data_path} for {epochs} epochs...")
    try:
        subprocess.check_call([
            sys.executable, "train_model.py", 
            "--data", data_path,
            "--epochs", str(epochs)
        ])
        print("Model training completed.")
        return True
    except subprocess.CalledProcessError:
        print("Failed to train model.")
        return False

def run_app():
    """Run the Streamlit application."""
    print("Starting Streamlit application...")
    try:
        subprocess.check_call(["streamlit", "run", "app.py"])
        return True
    except subprocess.CalledProcessError:
        print("Failed to run Streamlit application.")
        return False
    except FileNotFoundError:
        print("Streamlit not found. Make sure it's installed.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Project Runner")
    parser.add_argument("--setup", action="store_true", help="Set up the project")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data", type=str, default="samples", help="Path to training data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--run", action="store_true", help="Run the Streamlit app")
    parser.add_argument("--all", action="store_true", help="Do everything: setup, train, and run")
    
    args = parser.parse_args()
    
    # If no arguments are given, show help
    if not any([args.setup, args.train, args.run, args.all]):
        parser.print_help()
        return
    
    # Check requirements regardless of what we're doing
    if not check_requirements():
        print("Please install the required packages and try again.")
        return
    
    # Setup if requested
    if args.setup or args.all:
        if not setup_project():
            print("Project setup failed. Please check the errors and try again.")
            return
    
    # Train if requested
    if args.train or args.all:
        if not os.path.exists(args.data):
            print(f"Data directory {args.data} does not exist.")
            if args.all:
                print("Downloading sample data for training...")
                subprocess.check_call([sys.executable, "setup.py", "--download-samples"])
        
        if not train_model(args.data, args.epochs):
            print("Model training failed. Please check the errors and try again.")
            if not args.all:
                return
    
    # Run if requested
    if args.run or args.all:
        if not os.path.exists("models/feature_network.h5"):
            print("Model file not found. Please train the model first.")
            return
        
        run_app()

if __name__ == "__main__":
    main()