"""
MeowTrix-AI Feature Extractor Module
Implements Local Binary Patterns (LBP) and Histogram of Oriented Gradients (HOG) 
for texture and edge feature extraction from face images
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage import feature
import logging
from typing import Tuple, Optional, Union

class FeatureExtractor:
    """
    Feature extraction class implementing LBP and HOG descriptors
    for deepfake detection in face images
    """

    def __init__(self, 
                 lbp_radius: int = 1, 
                 lbp_n_points: int = 8,
                 hog_orientations: int = 9,
                 hog_pixels_per_cell: Tuple[int, int] = (16, 16),
                 hog_cells_per_block: Tuple[int, int] = (2, 2)):
        """
        Initialize feature extractor with LBP and HOG parameters

        Args:
            lbp_radius: Radius of LBP sampling
            lbp_n_points: Number of sampling points for LBP
            hog_orientations: Number of orientation bins for HOG
            hog_pixels_per_cell: Size of each cell for HOG
            hog_cells_per_block: Number of cells per block for HOG
        """
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block

        self.logger = self._setup_logger()

        # Calculate expected feature dimensions
        self.lbp_bins = self.lbp_n_points + 2  # uniform LBP bins
        self.hog_feature_length = self._calculate_hog_feature_length()

        self.logger.info("Feature Extractor initialized")
        self.logger.info(f"LBP parameters: radius={lbp_radius}, n_points={lbp_n_points}")
        self.logger.info(f"HOG parameters: orientations={hog_orientations}, "
                        f"pixels_per_cell={hog_pixels_per_cell}, cells_per_block={hog_cells_per_block}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the feature extractor"""
        logger = logging.getLogger('MeowTrix.FeatureExtractor')
        logger.setLevel(logging.INFO)
        return logger

    def _calculate_hog_feature_length(self) -> int:
        """Calculate expected HOG feature vector length"""
        # This is an approximation - actual length depends on image size
        # For 128x128 image with default parameters: approximately 1764 features
        return 1764

    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Pattern features from image

        Args:
            image: Grayscale image as numpy array

        Returns:
            LBP histogram feature vector
        """
        try:
            # Ensure image is 2D grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Extract LBP with uniform patterns
            lbp = local_binary_pattern(image, 
                                     self.lbp_n_points, 
                                     self.lbp_radius, 
                                     method='uniform')

            # Calculate histogram of LBP codes
            lbp_histogram, _ = np.histogram(lbp.ravel(), 
                                          bins=self.lbp_bins,
                                          range=(0, self.lbp_bins),
                                          density=True)

            self.logger.debug(f"LBP feature extraction completed, feature size: {len(lbp_histogram)}")
            return lbp_histogram.astype(np.float32)

        except Exception as e:
            self.logger.error(f"LBP feature extraction failed: {str(e)}")
            raise

    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients features from image

        Args:
            image: Grayscale image as numpy array

        Returns:
            HOG feature vector
        """
        try:
            # Ensure image is 2D grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Extract HOG features
            hog_features = hog(image,
                              orientations=self.hog_orientations,
                              pixels_per_cell=self.hog_pixels_per_cell,
                              cells_per_block=self.hog_cells_per_block,
                              block_norm='L2-Hys',
                              visualize=False,
                              feature_vector=True)

            self.logger.debug(f"HOG feature extraction completed, feature size: {len(hog_features)}")
            return hog_features.astype(np.float32)

        except Exception as e:
            self.logger.error(f"HOG feature extraction failed: {str(e)}")
            raise

    def extract_combined_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract both LBP and HOG features and concatenate them

        Args:
            image: Input grayscale image

        Returns:
            Combined feature vector (LBP + HOG)
        """
        try:
            # Extract individual features
            lbp_features = self.extract_lbp_features(image)
            hog_features = self.extract_hog_features(image)

            # Concatenate features
            combined_features = np.concatenate([lbp_features, hog_features])

            self.logger.info(f"Combined feature extraction completed: "
                           f"LBP({len(lbp_features)}) + HOG({len(hog_features)}) = "
                           f"Total({len(combined_features)})")

            return combined_features

        except Exception as e:
            self.logger.error(f"Combined feature extraction failed: {str(e)}")
            raise

    def extract_multi_scale_lbp(self, image: np.ndarray, scales: list = [1, 2, 3]) -> np.ndarray:
        """
        Extract LBP features at multiple scales for robustness

        Args:
            image: Input grayscale image
            scales: List of radius scales to use

        Returns:
            Multi-scale LBP feature vector
        """
        try:
            all_features = []

            for scale in scales:
                # Calculate LBP at current scale
                radius = self.lbp_radius * scale
                n_points = self.lbp_n_points * scale

                lbp = local_binary_pattern(image, n_points, radius, method='uniform')

                # Calculate histogram
                bins = n_points + 2
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)

                all_features.append(lbp_hist)

            # Concatenate all scale features
            multi_scale_features = np.concatenate(all_features)

            self.logger.debug(f"Multi-scale LBP extraction completed, total features: {len(multi_scale_features)}")
            return multi_scale_features.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Multi-scale LBP extraction failed: {str(e)}")
            raise

    def extract_texture_statistics(self, image: np.ndarray) -> np.ndarray:
        """
        Extract additional texture statistics features

        Args:
            image: Input grayscale image

        Returns:
            Texture statistics feature vector
        """
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            features = []

            # Basic statistical features
            features.extend([
                np.mean(image),      # Mean intensity
                np.std(image),       # Standard deviation
                np.var(image),       # Variance
                np.min(image),       # Minimum intensity
                np.max(image),       # Maximum intensity
            ])

            # Gradient magnitude statistics
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude)
            ])

            # Edge density
            edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)

            texture_features = np.array(features, dtype=np.float32)

            self.logger.debug(f"Texture statistics extraction completed: {len(texture_features)} features")
            return texture_features

        except Exception as e:
            self.logger.error(f"Texture statistics extraction failed: {str(e)}")
            raise

    def extract_all_features(self, image: np.ndarray) -> dict:
        """
        Extract all available features from image

        Args:
            image: Input image

        Returns:
            Dictionary containing all feature types
        """
        try:
            features = {
                'lbp': self.extract_lbp_features(image),
                'hog': self.extract_hog_features(image),
                'combined': self.extract_combined_features(image),
                'multi_scale_lbp': self.extract_multi_scale_lbp(image),
                'texture_stats': self.extract_texture_statistics(image)
            }

            # Log feature dimensions
            for feature_name, feature_vector in features.items():
                self.logger.info(f"{feature_name}: {len(feature_vector)} dimensions")

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def visualize_lbp(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize LBP pattern on image

        Args:
            image: Input grayscale image

        Returns:
            Tuple of (LBP pattern image, LBP histogram)
        """
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            lbp = local_binary_pattern(image, self.lbp_n_points, self.lbp_radius, method='uniform')

            # Normalize LBP for visualization
            lbp_normalized = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)

            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_bins, range=(0, self.lbp_bins))

            return lbp_normalized, hist

        except Exception as e:
            self.logger.error(f"LBP visualization failed: {str(e)}")
            raise

    def get_feature_info(self) -> dict:
        """
        Get information about feature extractor configuration

        Returns:
            Dictionary with feature extractor information
        """
        return {
            'lbp': {
                'radius': self.lbp_radius,
                'n_points': self.lbp_n_points,
                'expected_features': self.lbp_bins
            },
            'hog': {
                'orientations': self.hog_orientations,
                'pixels_per_cell': self.hog_pixels_per_cell,
                'cells_per_block': self.hog_cells_per_block,
                'expected_features': self.hog_feature_length
            }
        }
