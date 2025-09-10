"""
MeowTrix-AI Image Processor Module
Handles image loading, preprocessing, resizing, grayscale conversion, and normalization
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
import os
from typing import Tuple, Union, Optional
import logging

class ImageProcessor:
    """
    Image preprocessing class for MeowTrix-AI deepfake detection system.
    Handles loading, resizing, grayscale conversion, and normalization.
    """

    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the Image Processor

        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the image processor"""
        logger = logging.getLogger('MeowTrix.ImageProcessor')
        logger.setLevel(logging.INFO)
        return logger

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file path

        Args:
            image_path: Path to the image file

        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None

            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            self.logger.info(f"Successfully loaded image: {image_path}")
            return image

        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        
     try:
        # Step 1: Resize image
        resized_image = self.resize_image(image, self.target_size)

        # Step 2: Convert to grayscale
        gray_image = self.convert_to_grayscale(resized_image)

        # No normalization here!
        self.logger.info("Image preprocessing completed successfully")
        return gray_image  # dtype: uint8

     except Exception as e:
        self.logger.error(f"Preprocessing failed: {str(e)}")
        raise


    def resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target dimensions

        Args:
            image: Input image
            size: Target size (width, height)

        Returns:
            Resized image
        """
        return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale

        Args:
            image: Input BGR/RGB image

        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0

    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in image using Haar cascades

        Args:
            image: Input image

        Returns:
            Face bounding box (x, y, w, h) or None if no face detected
        """
        try:
            # Load Haar cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Convert to grayscale if needed
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Return the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                return tuple(largest_face)

            return None

        except Exception as e:
            self.logger.error(f"Face detection failed: {str(e)}")
            return None

    def crop_face(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop face from image using bounding box coordinates

        Args:
            image: Input image
            face_coords: Face bounding box (x, y, w, h)

        Returns:
            Cropped face image
        """
        x, y, w, h = face_coords
        return image[y:y+h, x:x+w]

    def batch_process_images(self, image_paths: list, output_dir: str) -> int:
        """
        Process multiple images in batch

        Args:
            image_paths: List of image file paths
            output_dir: Directory to save processed images

        Returns:
            Number of successfully processed images
        """
        os.makedirs(output_dir, exist_ok=True)
        success_count = 0

        for i, image_path in enumerate(image_paths):
            try:
                # Load and preprocess image
                image = self.load_image(image_path)
                if image is None:
                    continue

                processed_image = self.preprocess_image(image)

                # Save processed image
                filename = f"processed_{i:04d}.jpg"
                output_path = os.path.join(output_dir, filename)

                # Convert back to 0-255 range for saving
                save_image = (processed_image * 255).astype(np.uint8)
                cv2.imwrite(output_path, save_image)

                success_count += 1

            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {str(e)}")
                continue

        self.logger.info(f"Batch processing completed: {success_count}/{len(image_paths)} images")
        return success_count
