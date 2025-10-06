"""
Cairo Q-Lattice Analysis Module
================================

Implements topological data analysis tools for detecting Cairo pentagonal 
tiling patterns in KRAM fields and CMB maps.

The Cairo Q-Lattice is characterized by:
    - Pentagonal (5-fold local) symmetry
    - Alternating 3-valent and 4-valent vertices
    - Specific angular distributions (72°, 108° angles)
    - Aperiodic but highly ordered structure

This module provides:
    1. Pattern detection in 2D scalar fields
    2. Topological feature extraction (persistent homology)
    3. Statistical tests against null hypothesis (Gaussian random fields)
    4. Vertex configuration analysis
    5. Angular power spectrum methods

Author: David Noel Lynch
Date: 2025
License: MIT
"""

import numpy as np
from scipy import ndimage, signal, spatial, stats
from scipy.optimize import minimize
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import warnings


@dataclass
class CairoSignature:
    """
    Quantitative signature of Cairo Q-Lattice structure.
    
    Attributes:
        five_fold_power: Strength of 5-fold rotational symmetry
        vertex_3_4_ratio: Ratio of 3-valent to 4-valent vertices
        characteristic_angles: Presence of 72° and 108° angles
        spatial_periodicity: Dominant spatial frequencies
        topology_score: Persistent homology signature
        significance: Statistical significance vs. random field
    """
    five_fold_power: float
    vertex_3_4_ratio: float
    characteristic_angles: Tuple[float, float]
    spatial_periodicity: float
    topology_score: float
    significance: float
    
    def is_cairo_like(self, threshold: float = 0.7) -> bool:
        """
        Determine if signature indicates Cairo structure.
        
        Args:
            threshold: Confidence threshold [0, 1]
            
        Returns:
            True if Cairo-like pattern detected
        """
        # Weighted combination of indicators
        indicators = [
            self.five_fold_power > 0.3,
            0.6 < self.vertex_3_4_ratio < 1.0,
            self.characteristic_angles[0] > 0.5,  # 72° presence
            self.characteristic_angles[1] > 0.5,  # 108° presence
            self.significance > 3.0  # 3-sigma detection
        ]
        
        score = sum(indicators) / len(indicators)
        return score >= threshold


class PentagonDetector:
    """
    Detects pentagonal structures in 2D scalar fields using template matching.
    """
    
    def __init__(self, scale_range: Tuple[float, float] = (5.0, 20.0)):
        """
        Initialize pentagon detector.
        
        Args:
            scale_range: Range of pentagon sizes to search (min, max)
        """
        self.scale_range = scale_range
        self.templates = self._generate_templates()
    
    def _generate_templates(self) -> Dict[float, np.ndarray]:
        """
        Generate pentagon templates at different scales.
        
        Returns:
            Dictionary mapping scale → template array
        """
        templates = {}
        
        # Logarithmic scale sampling
        scales = np.geomspace(self.scale_range[0], self.scale_range[1], 8)
        
        for scale in scales:
            templates[scale] = self._create_pentagon_template(scale)
        
        return templates
    
    def _create_pentagon_template(self, radius: float, size: int = 64) -> np.ndarray:
        """
        Create regular pentagon template.
        
        Args:
            radius: Pentagon circumradius
            size: Template grid size
            
        Returns:
            Binary pentagon mask
        """
        center = size / 2
        y, x = np.ogrid[:size, :size]
        
        # Pentagon vertices
        angles = np.linspace(0, 2*np.pi, 6)  # 5 vertices + close
        vertices = []
        for angle in angles[:-1]:
            vx = center + radius * np.cos(angle - np.pi/2)
            vy = center + radius * np.sin(angle - np.pi/2)
            vertices.append([vx, vy])
        
        vertices = np.array(vertices)
        
        # Create mask using point-in-polygon test
        from matplotlib.path import Path
        path = Path(vertices)
        
        points = np.column_stack([x.ravel(), y.ravel()])
        mask = path.contains_points(points).reshape(size, size)
        
        # Apply Gaussian smoothing for soft edges
        mask = ndimage.gaussian_filter(mask.astype(float), sigma=1.0)
        
        # Normalize
        mask = mask / np.max(mask)
        
        return mask
    
    def detect(self, field: np.ndarray, threshold: float = 0.6) -> List[Tuple[int, int, float]]:
        """
        Detect pentagons in field using template matching.
        
        Args:
            field: 2D scalar field to analyze
            threshold: Detection threshold [0, 1]
            
        Returns:
            List of (y, x, scale) tuples for detected pentagons
        """
        detections = []
        
        # Normalize field
        field_norm = (field - np.mean(field)) / (np.std(field) + 1e-10)
        
        for scale, template in self.templates.items():
            # Cross-correlation
            correlation = signal.correlate2d(field_norm, template, mode='same', boundary='wrap')
            
            # Normalize correlation
            correlation = correlation / np.max(np.abs(correlation))
            
            # Find local maxima above threshold
            local_max = (correlation == ndimage.maximum_filter(correlation, size=int(scale)))
            peaks = np.where((local_max) & (correlation > threshold))
            
            for y, x in zip(peaks[0], peaks[1]):
                detections.append((y, x, scale, correlation[y, x]))
        
        # Non-maximum suppression across scales
        detections = self._non_maximum_suppression(detections)
        
        return detections
    
    def _non_maximum_suppression(self, detections: List, radius: float = 15.0) -> List:
        """
        Remove overlapping detections, keeping highest score.
        
        Args:
            detections: List of (y, x, scale, score)
            radius: Suppression radius
            
        Returns:
            Filtered detections
        """
        if not detections:
            return []
        
        # Sort by score descending
        detections = sorted(detections, key=lambda d: d[3], reverse=True)
        
        kept = []
        for det in detections:
            y, x, scale, score = det
            
            # Check if too close to already kept detection
            too_close = False
            for kept_det in kept:
                ky, kx = kept_det[0], kept_det[1]
                dist = np.sqrt((y - ky)**2 + (x - kx)**2)
                if dist < radius:
                    too_close = True
                    break
            
            if not too_close:
                kept.append(det)
        
        return kept


class VertexAnalyzer:
    """
    Analyzes vertex configurations (valence distribution) in field extrema.
    
    Cairo tiling has characteristic 3-valent and 4-valent vertices.
    """
    
    def __init__(self, connection_radius: float = 3.0):
        """
        Initialize vertex analyzer.
        
        Args:
            connection_radius: Distance threshold for vertex connections
        """
        self.connection_radius = connection_radius
    
    def extract_extrema(self, field: np.ndarray, percentile: float = 90) -> np.ndarray:
        """
        Extract local extrema (peaks and valleys) as vertices.
        
        Args:
            field: 2D scalar field
            percentile: Percentile threshold for extrema
            
        Returns:
            Array of (y, x) coordinates
        """
        # Find local maxima
        max_filter = ndimage.maximum_filter(field, size=3)
        maxima = (field == max_filter) & (field > np.percentile(field, percentile))
        
        # Find local minima
        min_filter = ndimage.minimum_filter(field, size=3)
        minima = (field == min_filter) & (field < np.percentile(field, 100 - percentile))
        
        # Combine
        extrema = maxima | minima
        
        # Extract coordinates
        coords = np.column_stack(np.where(extrema))
        
        return coords
    
    def compute_valences(self, vertices: np.ndarray) -> np.ndarray:
        """
        Compute valence (number of connections) for each vertex.
        
        Args:
            vertices: Array of (y, x) coordinates
            
        Returns:
            Array of valences
        """
        if len(vertices) == 0:
            return np.array([])
        
        # Build KD-tree for efficient nearest neighbor search
        tree = spatial.KDTree(vertices)
        
        # Find all neighbors within radius
        valences = []
        for vertex in vertices:
            neighbors = tree.query_ball_point(vertex, self.connection_radius)
            # Subtract 1 to exclude self
            valences.append(len(neighbors) - 1)
        
        return np.array(valences)
    
    def analyze_valence_distribution(self, field: np.ndarray) -> Tuple[Dict[int, int], float]:
        """
        Analyze vertex valence distribution.
        
        Args:
            field: 2D scalar field
            
        Returns:
            valence_counts: Dictionary mapping valence → count
            vertex_3_4_ratio: Ratio of 3-valent to 4-valent vertices
        """
        vertices = self.extract_extrema(field)
        valences = self.compute_valences(vertices)
        
        if len(valences) == 0:
            return {}, 0.0
        
        # Count valences
        unique, counts = np.unique(valences, return_counts=True)
        valence_counts = dict(zip(unique.astype(int), counts.astype(int)))
        
        # Cairo signature: approximately equal 3-valent and 4-valent vertices
        count_3 = valence_counts.get(3, 0)
        count_4 = valence_counts.get(4, 0)
        
        if count_3 + count_4 > 0:
            vertex_3_4_ratio = count_3 / (count_4 + 1e-10)
        else:
            vertex_3_4_ratio = 0.0
        
        return valence_counts, vertex_3_4_ratio


class AngleAnalyzer:
    """
    Analyzes angular distributions in field patterns.
    
    Cairo tiling has characteristic angles: 72° and 108° (from pentagons).
    """
    
    def __init__(self, n_bins: int = 72):
        """
        Initialize angle analyzer.
        
        Args:
            n_bins: Number of angular bins (should be multiple of 36 for 5-fold)
        """
        self.n_bins = n_bins
        self.bin_width = 360.0 / n_bins
    
    def compute_gradient_angles(self, field: np.ndarray) -> np.ndarray:
        """
        Compute gradient directions in field.
        
        Args:
            field: 2D scalar field
            
        Returns:
            Array of angles in degrees [0, 360)
        """
        # Compute gradients
        gy, gx = np.gradient(field)
        
        # Compute angles
        angles = np.arctan2(gy, gx)
        angles = np.degrees(angles) % 360
        
        return angles.ravel()
    
    def detect_characteristic_angles(self, field: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Detect presence of 72° and 108° angles (pentagonal signature).
        
        Args:
            field: 2D scalar field
            
        Returns:
            angle_72_power: Strength of 72° signal
            angle_108_power: Strength of 108° signal
            histogram: Full angular histogram
        """
        angles = self.compute_gradient_angles(field)
        
        # Create histogram
        hist, bin_edges = np.histogram(angles, bins=self.n_bins, range=(0, 360))
        
        # Normalize
        hist = hist / np.sum(hist)
        
        # Find peaks near characteristic angles
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 72° (pentagon interior angle / 5)
        idx_72 = np.argmin(np.abs(bin_centers - 72))
        angle_72_power = hist[idx_72]
        
        # 108° (pentagon interior angle)
        idx_108 = np.argmin(np.abs(bin_centers - 108))
        angle_108_power = hist[idx_108]
        
        # Also check at multiples (144°, 216°, etc.)
        multiples_72 = [72, 144, 216, 288]
        multiples_108 = [108, 216]
        
        total_72 = sum(hist[np.argmin(np.abs(bin_centers - angle))] for angle in multiples_72)
        total_108 = sum(hist[np.argmin(np.abs(bin_centers - angle))] for angle in multiples_108)
        
        return total_72, total_108, hist
    
    def compute_angular_power_spectrum(self, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum of angular distribution.
        
        Args:
            angles: Array of angles
            
        Returns:
            frequencies: Angular frequency modes
            power: Power at each frequency
        """
        # Histogram
        hist, _ = np.histogram(angles, bins=self.n_bins, range=(0, 360))
        
        # FFT
        fft = np.fft.fft(hist)
        power = np.abs(fft)**2
        
        # Frequencies (angular modes)
        frequencies = np.fft.fftfreq(self.n_bins, d=self.bin_width)
        
        return frequencies[:self.n_bins//2], power[:self.n_bins//2]


class FiveFoldSymmetryDetector:
    """
    Detects 5-fold rotational symmetry in 2D fields.
    
    Uses rotational autocorrelation and Fourier analysis.
    """
    
    def __init__(self, n_angles: int = 360):
        """
        Initialize symmetry detector.
        
        Args:
            n_angles: Number of rotation angles to test
        """
        self.n_angles = n_angles
        self.angles = np.linspace(0, 360, n_angles, endpoint=False)
    
    def compute_rotational_autocorrelation(self, field: np.ndarray) -> np.ndarray:
        """
        Compute autocorrelation as function of rotation angle.
        
        Args:
            field: 2D scalar field
            
        Returns:
            Autocorrelation array (one value per angle)
        """
        autocorr = np.zeros(self.n_angles)
        
        # Normalize field
        field_norm = field - np.mean(field)
        field_norm = field_norm / np.std(field_norm)
        
        for i, angle in enumerate(self.angles):
            # Rotate field
            rotated = ndimage.rotate(field_norm, angle, reshape=False, order=1)
            
            # Compute correlation
            autocorr[i] = np.corrcoef(field_norm.ravel(), rotated.ravel())[0, 1]
        
        return autocorr
    
    def detect_five_fold(self, field: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect 5-fold symmetry using rotational autocorrelation.
        
        Args:
            field: 2D scalar field
            
        Returns:
            five_fold_power: Strength of 5-fold symmetry [0, 1]
            autocorr: Full autocorrelation curve
        """
        autocorr = self.compute_rotational_autocorrelation(field)
        
        # FFT to find dominant angular frequencies
        fft = np.fft.fft(autocorr)
        power = np.abs(fft)**2
        
        # 5-fold symmetry: peak at frequency k=5
        # (autocorrelation repeats 5 times in 360°)
        freq_5 = 5
        five_fold_power = power[freq_5] / np.sum(power[1:self.n_angles//2])
        
        return five_fold_power, autocorr


class CairoLatticeAnalyzer:
    """
    Complete Cairo Q-Lattice analysis suite.
    
    Combines multiple detection methods to produce comprehensive signature.
    """
    
    def __init__(self):
        """Initialize all analysis components."""
        self.pentagon_detector = PentagonDetector()
        self.vertex_analyzer = VertexAnalyzer()
        self.angle_analyzer = AngleAnalyzer()
        self.symmetry_detector = FiveFoldSymmetryDetector()
    
    def analyze(self, field: np.ndarray, 
                compute_significance: bool = True,
                n_bootstrap: int = 100) -> CairoSignature:
        """
        Perform complete Cairo lattice analysis.
        
        Args:
            field: 2D scalar field to analyze
            compute_significance: Whether to compute statistical significance
            n_bootstrap: Number of bootstrap samples for significance
            
        Returns:
            CairoSignature object with all metrics
        """
        print("Analyzing field for Cairo Q-Lattice signatures...")
        
        # 1. Five-fold symmetry
        print("  - Computing 5-fold symmetry...")
        five_fold_power, _ = self.symmetry_detector.detect_five_fold(field)
        
        # 2. Vertex configuration
        print("  - Analyzing vertex configurations...")
        valence_dist, vertex_3_4_ratio = self.vertex_analyzer.analyze_valence_distribution(field)
        
        # 3. Characteristic angles
        print("  - Detecting characteristic angles...")
        angle_72, angle_108, _ = self.angle_analyzer.detect_characteristic_angles(field)
        
        # 4. Spatial periodicity (from power spectrum)
        print("  - Computing spatial periodicity...")
        fft = np.fft.fft2(field)
        power_2d = np.abs(fft)**2
        
        # Radial average
        center = np.array(field.shape) // 2
        y, x = np.indices(field.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        r_int = r.astype(int)
        radial_profile = ndimage.mean(power_2d, labels=r_int, index=np.arange(r_int.max() + 1))
        
        # Find dominant frequency
        peak_idx = np.argmax(radial_profile[1:]) + 1  # Skip DC
        spatial_periodicity = peak_idx
        
        # 5. Simple topology score (peak count as proxy)
        topology_score = len(self.pentagon_detector.detect(field, threshold=0.5))
        
        # 6. Statistical significance
        if compute_significance:
            print("  - Computing statistical significance...")
            significance = self._compute_significance(field, n_bootstrap)
        else:
            significance = 0.0
        
        signature = CairoSignature(
            five_fold_power=five_fold_power,
            vertex_3_4_ratio=vertex_3_4_ratio,
            characteristic_angles=(angle_72, angle_108),
            spatial_periodicity=spatial_periodicity,
            topology_score=topology_score,
            significance=significance
        )
        
        print(f"\nResults:")
        print(f"  5-fold power: {five_fold_power:.4f}")
        print(f"  3/4-vertex ratio: {vertex_3_4_ratio:.3f}")
        print(f"  72° angle power: {angle_72:.4f}")
        print(f"  108° angle power: {angle_108:.4f}")
        print(f"  Spatial period: {spatial_periodicity:.1f} pixels")
        print(f"  Significance: {significance:.2f}σ")
        print(f"  Cairo-like: {signature.is_cairo_like()}")
        
        return signature
    
    def _compute_significance(self, field: np.ndarray, n_bootstrap: int) -> float:
        """
        Compute statistical significance vs. Gaussian random field null hypothesis.
        
        Args:
            field: Observed field
            n_bootstrap: Number of random realizations
            
        Returns:
            Significance in standard deviations (σ)
        """
        # Compute observed five-fold power (main Cairo indicator)
        obs_power, _ = self.symmetry_detector.detect_five_fold(field)
        
        # Generate bootstrap samples from Gaussian random field with same power spectrum
        fft_obs = np.fft.fft2(field)
        power_spectrum = np.abs(fft_obs)
        
        bootstrap_powers = []
        for _ in range(n_bootstrap):
            # Random phases
            random_phases = np.exp(2j * np.pi * np.random.rand(*field.shape))
            fft_random = power_spectrum * random_phases
            field_random = np.real(np.fft.ifft2(fft_random))
            
            # Compute five-fold power
            power, _ = self.symmetry_detector.detect_five_fold(field_random)
            bootstrap_powers.append(power)
        
        bootstrap_powers = np.array(bootstrap_powers)
        
        # Compute z-score
        mean_null = np.mean(bootstrap_powers)
        std_null = np.std(bootstrap_powers)
        
        if std_null > 0:
            z_score = (obs_power - mean_null) / std_null
        else:
            z_score = 0.0
        
        return z_score


# ============================================================================
# CMB Map Analysis
# ============================================================================

def analyze_cmb_map(temperature_map: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> CairoSignature:
    """
    Analyze CMB temperature map for Cairo Q-Lattice signatures.
    
    Args:
        temperature_map: 2D CMB temperature map (e.g., Planck data)
        mask: Optional mask (True = good pixels, False = masked)
        
    Returns:
        CairoSignature from CMB analysis
    """
    if mask is not None:
        # Apply mask
        map_masked = temperature_map.copy()
        map_masked[~mask] = np.nan
        
        # Interpolate over masked regions for analysis
        from scipy.interpolate import griddata
        valid = ~np.isnan(map_masked)
        coords_valid = np.column_stack(np.where(valid))
        values_valid = map_masked[valid]
        
        coords_all = np.column_stack(np.where(np.ones_like(map_masked, dtype=bool)))
        map_interp = griddata(coords_valid, values_valid, coords_all, method='cubic')
        map_interp = map_interp.reshape(map_masked.shape)
        
        field = map_interp
    else:
        field = temperature_map
    
    # Run full analysis
    analyzer = CairoLatticeAnalyzer()
    signature = analyzer.analyze(field, compute_significance=True, n_bootstrap=50)
    
    return signature


# ============================================================================
# Example Usage
# ============================================================================

def example_synthetic_cairo_field():
    """Example: Analyze synthetic field with Cairo-like properties."""
    print("Example 1: Synthetic Cairo-Like Field")
    print("=" * 70)
    
    # Generate synthetic field with pentagonal features
    size = 128
    field = np.zeros((size, size))
    
    # Add pentagon-like features at regular intervals
    spacing = 20
    for i in range(3, size, spacing):
        for j in range(3, size, spacing):
            # Pentagon-like radial pattern
            y, x = np.ogrid[:size, :size]
            r = np.sqrt((x - j)**2 + (y - i)**2)
            theta = np.arctan2(y - i, x - j)
            
            # 5-fold modulation
            pattern = np.cos(5 * theta) * np.exp(-r**2 / 50)
            field += pattern
    
    # Add noise
    field += np.random.randn(size, size) * 0.5
    
    # Analyze
    analyzer = CairoLatticeAnalyzer()
    signature = analyzer.analyze(field, compute_significance=True, n_bootstrap=30)
    
    print(f"\nIs Cairo-like: {signature.is_cairo_like()}")
    
    return field, signature


def example_random_gaussian_field():
    """Example: Analyze purely random field (null hypothesis)."""
    print("\nExample 2: Random Gaussian Field (Null Hypothesis)")
    print("=" * 70)
    
    # Generate random field
    size = 128
    field = np.random.randn(size, size)
    
    # Apply smoothing to create spatial correlation
    field = ndimage.gaussian_filter(field, sigma=3.0)
    
    # Analyze
    analyzer = CairoLatticeAnalyzer()
    signature = analyzer.analyze(field, compute_significance=True, n_bootstrap=30)
    
    print(f"\nIs Cairo-like: {signature.is_cairo_like()}")
    
    return field, signature


def example_kram_output_analysis():
    """Example: Analyze KRAM evolution output."""
    print("\nExample 3: KRAM Evolution Output")
    print("=" * 70)
    
    # Simulate KRAM-like field with hexagonal structure
    size = 128
    x = np.linspace(0, 10*np.pi, size)
    y = np.linspace(0, 10*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # Hexagonal pattern (approximation to Cairo)
    k = 1.0
    field = (np.cos(k * X) + 
            np.cos(k * (0.5 * X - np.sqrt(3)/2 * Y)) +
            np.cos(k * (0.5 * X + np.sqrt(3)/2 * Y)))
    
    # Add higher harmonics and noise
    field += 0.3 * np.cos(2 * k * X) * np.sin(2 * k * Y)
    field += np.random.randn(size, size) * 0.2
    
    # Analyze
    analyzer = CairoLatticeAnalyzer()
    signature = analyzer.analyze(field, compute_significance=True, n_bootstrap=30)
    
    print(f"\nIs Cairo-like: {signature.is_cairo_like()}")
    
    return field, signature


if __name__ == "__main__":
    print("=" * 70)
    print("Cairo Q-Lattice Analysis Module - Test Suite")
    print("=" * 70)
    print()
    
    # Run examples
    field1, sig1 = example_synthetic_cairo_field()
    field2, sig2 = example_random_gaussian_field()
    field3, sig3 = example_kram_output_analysis()
    
    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)
    print(f"Synthetic Cairo:  Detected = {sig1.is_cairo_like()}, σ = {sig1.significance:.2f}")
    print(f"Random Gaussian:  Detected = {sig2.is_cairo_like()}, σ = {sig2.significance:.2f}")
    print(f"KRAM Hexagonal:   Detected = {sig3.is_cairo_like()}, σ = {sig3.significance:.2f}")
    print("\n✓ Cairo analysis pipeline operational")
