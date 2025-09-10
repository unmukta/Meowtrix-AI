"""
MeowTrix-AI Interface Module
Provides both Command Line Interface (CLI) and Graphical User Interface (GUI)
for deepfake detection system
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import argparse
import sys
import os
import numpy as np
from PIL import Image, ImageTk
import cv2
import threading
import logging
from typing import Optional, Dict, Any
import json

# Import MeowTrix modules
from image_processor import ImageProcessor
from feature_extractor import FeatureExtractor
from classifier import MeowTrixClassifier

class MeowTrixCLI:
    """
    Command Line Interface for MeowTrix-AI deepfake detection
    """

    def __init__(self):
        self.image_processor = ImageProcessor()
        self.feature_extractor = FeatureExtractor()
        self.classifier = MeowTrixClassifier()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for CLI"""
        logger = logging.getLogger('MeowTrix.CLI')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def detect_single_image(self, image_path: str, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect deepfake in a single image

        Args:
            image_path: Path to the image file
            model_path: Path to trained model file

        Returns:
            Detection results dictionary
        """
        try:
            self.logger.info(f"Processing image: {image_path}")

            # Load model if provided
            if model_path and os.path.exists(model_path):
                self.classifier.load_model(model_path)
            elif not self.classifier.is_fitted:
                raise ValueError("No trained model available. Please provide a model path or train a model.")

            # Load and preprocess image
            image = self.image_processor.load_image(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            processed_image = self.image_processor.preprocess_image(image)

            # Extract features
            features = self.feature_extractor.extract_combined_features(processed_image)
            features = features.reshape(1, -1)  # Reshape for single prediction

            # Make prediction
            prediction = self.classifier.predict(features)[0]
            probabilities = self.classifier.predict_proba(features)[0]

            # Prepare results
            result = {
                'image_path': image_path,
                'prediction': 'Real' if prediction == 1 else 'Fake',
                'confidence': float(max(probabilities)),
                'fake_probability': float(probabilities[0]),
                'real_probability': float(probabilities[1])
            }

            self.logger.info(f"Detection result: {result['prediction']} ({result['confidence']:.2%} confidence)")
            return result

        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            raise

    def batch_detect(self, input_dir: str, output_file: str, model_path: Optional[str] = None) -> None:
        """
        Detect deepfakes in a batch of images

        Args:
            input_dir: Directory containing images
            output_file: CSV file to save results
            model_path: Path to trained model file
        """
        try:
            # Load model
            if model_path and os.path.exists(model_path):
                self.classifier.load_model(model_path)
            elif not self.classifier.is_fitted:
                raise ValueError("No trained model available")

            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = []

            for file in os.listdir(input_dir):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(input_dir, file))

            self.logger.info(f"Processing {len(image_files)} images...")

            results = []
            for i, image_path in enumerate(image_files, 1):
                try:
                    result = self.detect_single_image(image_path)
                    results.append(result)
                    self.logger.info(f"Progress: {i}/{len(image_files)}")

                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {str(e)}")
                    continue

            # Save results to CSV
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            self.logger.info(f"Results saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Batch detection failed: {str(e)}")
            raise

    @staticmethod
    def main():
        """Main CLI entry point"""
        parser = argparse.ArgumentParser(description='MeowTrix-AI Deepfake Detection CLI')
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Single detection command
        detect_parser = subparsers.add_parser('detect', help='Detect deepfake in single image')
        detect_parser.add_argument('image', help='Path to image file')
        detect_parser.add_argument('--model', help='Path to trained model file')

        # Batch detection command
        batch_parser = subparsers.add_parser('batch', help='Batch detection on directory')
        batch_parser.add_argument('input_dir', help='Directory containing images')
        batch_parser.add_argument('output', help='Output CSV file')
        batch_parser.add_argument('--model', help='Path to trained model file')

        # Train command
        train_parser = subparsers.add_parser('train', help='Train new model')
        train_parser.add_argument('real_dir', help='Directory containing real face images')
        train_parser.add_argument('fake_dir', help='Directory containing fake face images')
        train_parser.add_argument('output_model', help='Output model file path')
        train_parser.add_argument('--config', help='Configuration file path')

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        cli = MeowTrixCLI()

        try:
            if args.command == 'detect':
                result = cli.detect_single_image(args.image, args.model)
                print(f"\nResult: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Fake Probability: {result['fake_probability']:.2%}")
                print(f"Real Probability: {result['real_probability']:.2%}")

            elif args.command == 'batch':
                cli.batch_detect(args.input_dir, args.output, args.model)

            elif args.command == 'train':
                print("Training functionality would be implemented here")
                print("This would involve loading datasets, training models, and saving results")

        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)


class MeowTrixGUI:
    """
    Graphical User Interface for MeowTrix-AI deepfake detection
    """

    def __init__(self, root):
        self.root = root
        self.root.title("MeowTrix-AI - Deepfake Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        # Initialize components
        self.image_processor = ImageProcessor()
        self.feature_extractor = FeatureExtractor()
        self.classifier = MeowTrixClassifier()

        # GUI variables
        self.current_image = None
        self.current_image_path = None
        self.model_loaded = False

        # Create GUI elements
        self.setup_gui()

        # Setup logging
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for GUI"""
        logger = logging.getLogger('MeowTrix.GUI')
        logger.setLevel(logging.INFO)
        return logger

    def setup_gui(self):
        """Setup the GUI layout"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="MeowTrix-AI Deepfake Detection", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # Load model button
        self.load_model_btn = ttk.Button(control_frame, text="Load Model", 
                                        command=self.load_model)
        self.load_model_btn.grid(row=0, column=0, padx=(0, 10))

        # Load image button
        self.load_image_btn = ttk.Button(control_frame, text="Select Image", 
                                        command=self.load_image)
        self.load_image_btn.grid(row=0, column=1, padx=(0, 10))

        # Analyze button
        self.analyze_btn = ttk.Button(control_frame, text="Analyze Image", 
                                     command=self.analyze_image, state=tk.DISABLED)
        self.analyze_btn.grid(row=0, column=2, padx=(0, 10))

        # Batch processing button
        self.batch_btn = ttk.Button(control_frame, text="Batch Process", 
                                   command=self.batch_process, state=tk.DISABLED)
        self.batch_btn.grid(row=0, column=3)

        # Image display area
        image_frame = ttk.LabelFrame(main_frame, text="Image", padding="10")
        image_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(image_frame, bg='white', width=300, height=300)
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)

        # Result display
        self.result_text = tk.Text(results_frame, height=15, width=40, state=tk.DISABLED)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please load a model first")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, font=('Arial', 9))
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

    def load_model(self):
        """Load a trained model file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("Joblib files", "*.joblib"), ("Pickle files", "*.pkl"), ("All files", "*.*")]
            )

            if file_path:
                self.status_var.set("Loading model...")
                self.root.update()

                # Load model in separate thread to prevent GUI freezing
                def load_model_thread():
                    try:
                        self.classifier.load_model(file_path)
                        self.model_loaded = True

                        # Update GUI in main thread
                        self.root.after(0, lambda: self.on_model_loaded(file_path))

                    except Exception as e:
                        self.root.after(0, lambda: self.on_model_load_error(str(e)))

                threading.Thread(target=load_model_thread, daemon=True).start()

        except Exception as e:
            self.show_error("Model Loading Error", f"Failed to load model: {str(e)}")

    def on_model_loaded(self, file_path):
        """Called when model is successfully loaded"""
        self.batch_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Model loaded: {os.path.basename(file_path)}")
        self.update_analyze_button_state()

        # Update results display
        self.update_results_display(f"✅ Model loaded successfully\n"
                                   f"Model file: {os.path.basename(file_path)}\n"
                                   f"Model type: {self.classifier.best_model_name}\n\n")

    def on_model_load_error(self, error_msg):
        """Called when model loading fails"""
        self.show_error("Model Loading Error", f"Failed to load model: {error_msg}")
        self.status_var.set("Ready - Please load a model")

    def load_image(self):
        """Load an image for analysis"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
            )

            if file_path:
                self.current_image_path = file_path
                self.display_image(file_path)
                self.update_analyze_button_state()

                # Update results display
                self.update_results_display(f"📷 Image loaded: {os.path.basename(file_path)}\n\n")

        except Exception as e:
            self.show_error("Image Loading Error", f"Failed to load image: {str(e)}")

    def display_image(self, image_path):
        """Display image in the canvas"""
        try:
            # Load and resize image for display
            pil_image = Image.open(image_path)

            # Calculate size to fit in canvas while maintaining aspect ratio
            canvas_width = self.image_canvas.winfo_width() or 300
            canvas_height = self.image_canvas.winfo_height() or 300

            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height)

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(display_image)

            # Clear canvas and display image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                         image=self.current_image)

        except Exception as e:
            self.show_error("Display Error", f"Failed to display image: {str(e)}")

    def update_analyze_button_state(self):
        """Update analyze button state based on loaded model and image"""
        if self.model_loaded and self.current_image_path:
            self.analyze_btn.configure(state=tk.NORMAL)
        else:
            self.analyze_btn.configure(state=tk.DISABLED)

    def analyze_image(self):
        """Analyze the loaded image for deepfake detection"""
        if not self.model_loaded or not self.current_image_path:
            self.show_error("Analysis Error", "Please load both a model and an image first")
            return

        try:
            self.status_var.set("Analyzing image...")
            self.analyze_btn.configure(state=tk.DISABLED)
            self.root.update()

            # Analyze in separate thread
            def analyze_thread():
                try:
                    # Load and preprocess image
                    image = self.image_processor.load_image(self.current_image_path)
                    processed_image = self.image_processor.preprocess_image(image)

                    # Extract features
                    features = self.feature_extractor.extract_combined_features(processed_image)
                    features = features.reshape(1, -1)

                    # Make prediction
                    prediction = self.classifier.predict(features)[0]
                    probabilities = self.classifier.predict_proba(features)[0]

                    # Prepare results
                    result = {
                        'prediction': 'Real' if prediction == 1 else 'Fake',
                        'confidence': float(max(probabilities)),
                        'fake_probability': float(probabilities[0]),
                        'real_probability': float(probabilities[1])
                    }

                    # Update GUI in main thread
                    self.root.after(0, lambda: self.on_analysis_complete(result))

                except Exception as e:
                    self.root.after(0, lambda: self.on_analysis_error(str(e)))

            threading.Thread(target=analyze_thread, daemon=True).start()

        except Exception as e:
            self.show_error("Analysis Error", f"Failed to analyze image: {str(e)}")
            self.analyze_btn.configure(state=tk.NORMAL)
            self.status_var.set("Ready")

    def on_analysis_complete(self, result):
        """Called when analysis is complete"""
        self.analyze_btn.configure(state=tk.NORMAL)
        self.status_var.set("Analysis complete")

        # Display results
        result_text = f"🔍 ANALYSIS RESULTS\n"
        result_text += f"{'='*30}\n\n"
        result_text += f"📊 Prediction: {result['prediction']}\n"
        result_text += f"🎯 Confidence: {result['confidence']:.2%}\n\n"
        result_text += f"📈 Detailed Probabilities:\n"
        result_text += f"  • Fake: {result['fake_probability']:.2%}\n"
        result_text += f"  • Real: {result['real_probability']:.2%}\n\n"

        if result['prediction'] == 'Fake':
            result_text += f"⚠️  WARNING: This image appears to be AI-generated or manipulated!\n\n"
        else:
            result_text += f"✅ This image appears to be authentic.\n\n"

        self.update_results_display(result_text)

    def on_analysis_error(self, error_msg):
        """Called when analysis fails"""
        self.analyze_btn.configure(state=tk.NORMAL)
        self.status_var.set("Analysis failed")
        self.show_error("Analysis Error", f"Failed to analyze image: {error_msg}")

    def batch_process(self):
        """Batch process multiple images"""
        if not self.model_loaded:
            self.show_error("Batch Processing Error", "Please load a model first")
            return

        try:
            input_dir = filedialog.askdirectory(title="Select Directory with Images")
            if not input_dir:
                return

            output_file = filedialog.asksaveasfilename(
                title="Save Results As",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not output_file:
                return

            self.status_var.set("Batch processing...")
            self.batch_btn.configure(state=tk.DISABLED)

            # Process in separate thread
            def batch_thread():
                try:
                    cli = MeowTrixCLI()
                    cli.classifier = self.classifier  # Use the loaded model
                    cli.batch_detect(input_dir, output_file)

                    self.root.after(0, lambda: self.on_batch_complete(output_file))

                except Exception as e:
                    self.root.after(0, lambda: self.on_batch_error(str(e)))

            threading.Thread(target=batch_thread, daemon=True).start()

        except Exception as e:
            self.show_error("Batch Processing Error", f"Failed to start batch processing: {str(e)}")

    def on_batch_complete(self, output_file):
        """Called when batch processing is complete"""
        self.batch_btn.configure(state=tk.NORMAL)
        self.status_var.set("Batch processing complete")

        messagebox.showinfo("Batch Processing Complete", 
                           f"Results saved to:\n{output_file}")

        self.update_results_display(f"📁 Batch processing completed\n"
                                   f"Results saved to: {os.path.basename(output_file)}\n\n")

    def on_batch_error(self, error_msg):
        """Called when batch processing fails"""
        self.batch_btn.configure(state=tk.NORMAL)
        self.status_var.set("Batch processing failed")
        self.show_error("Batch Processing Error", f"Batch processing failed: {error_msg}")

    def update_results_display(self, text):
        """Update the results text display"""
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)
        self.result_text.see(tk.END)

    def show_error(self, title, message):
        """Show error message dialog"""
        messagebox.showerror(title, message)

    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point for the interface module"""
    if len(sys.argv) > 1:
        # Run CLI if command line arguments are provided
        MeowTrixCLI.main()
    else:
        # Run GUI if no arguments provided
        root = tk.Tk()
        app = MeowTrixGUI(root)
        app.run()


if __name__ == "__main__":
    main()
