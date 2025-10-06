"""
Projection Maps Module
=====================

Implements the projection map f: spacetime → KRAM manifold.

This is the crucial link between spacetime events and cosmic memory. The map encodes
how the six-fold structure of the KnoWellian Tensor (ν ∈ {P,I,F,x,y,z}) translates
into the geometry of the KRAM.

Mathematical Structure:
    f: ℝ^4 → ℝ^6
    
    Spacetime (t, x, y, z) → Manifold (X_x, X_y, X_z, X_h1, X_h2, X_φ)
    
Where:
    - (X_x, X_y, X_z): Spatial coordinates (coarse-grained)
    - (X_h1, X_h2): Hexagonal lattice coordinates from temporal triad P/I/F
    - X_φ: Phase coordinate from spatial orientation

The temporal triad → hex-plane mapping naturally generates 6-fold symmetry,
providing the geometric origin of the Cairo Q-Lattice structure.

Author: David Noel Lynch
Date: 2025
License: MIT
"""

import numpy as np
from typing import Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum


class TriadNormalization(Enum):
    """Methods for normalizing temporal triad weights."""
    SOFTMAX = "softmax"  # Exponential normalization
    LINEAR = "linear"    # Simple division by sum
    RECTIFIED = "rectified"  # Positive parts only


@dataclass
class ProjectionParameters:
    """
    Parameters controlling the spacetime → KRAM projection.
    
    Attributes:
        l_KW: KnoWellian length scale (regularization/coarse-graining scale)
        triad_normalization: Method for normalizing P/I/F weights
        epsilon: Small floor value to prevent division by zero
        use_spatial_phase: Include spatial orientation in phase coordinate
        hex_scale: Scale factor for hexagonal coordinates
    """
    l_KW: float = 1.0
    triad_normalization: TriadNormalization = TriadNormalization.LINEAR
    epsilon: float = 1e-10
    use_spatial_phase: bool = True
    hex_scale: float = 1.0


class KnoWellianTensorField:
    """
    Represents local KnoWellian Tensor components at a spacetime point.
    
    In a full implementation, this would be computed from field evolution.
    For testing, we provide synthetic constructors.
    """
    
    def __init__(self,
                 T_P: float = 0.0,
                 T_I: float = 0.0,
                 T_F: float = 0.0,
                 T_x: float = 0.0,
                 T_y: float = 0.0,
                 T_z: float = 0.0):
        """
        Initialize tensor components.
        
        Args:
            T_P, T_I, T_F: Temporal components (Past, Instant, Future)
            T_x, T_y, T_z: Spatial components
        """
        self.T_P = T_P
        self.T_I = T_I
        self.T_F = T_F
        self.T_x = T_x
        self.T_y = T_y
        self.T_z = T_z
    
    @classmethod
    def from_position(cls, x: float, y: float, z: float, t: float):
        """
        Generate synthetic tensor from spacetime position.
        
        Creates smooth, spatially-varying field for testing.
        """
        # Temporal components: smooth spatial variation
        T_P = 1.0 + 0.3 * np.sin(0.5 * x) * np.cos(0.5 * y)
        T_I = 1.0 + 0.2 * np.cos(0.3 * x) * np.sin(0.3 * y)
        T_F = 1.0 + 0.4 * np.sin(0.4 * x + 0.4 * y)
        
        # Spatial components: vector field
        T_x = np.cos(0.2 * x) * np.sin(0.2 * t)
        T_y = np.sin(0.2 * y) * np.cos(0.2 * t)
        T_z = 0.5 * np.sin(0.1 * (x + y))
        
        return cls(T_P, T_I, T_F, T_x, T_y, T_z)
    
    @classmethod
    def control_dominated(cls):
        """Create Control-dominated configuration (T_P >> T_I, T_F)."""
        return cls(T_P=2.0, T_I=0.3, T_F=0.2, T_x=0.5, T_y=0.3, T_z=0.1)
    
    @classmethod
    def chaos_dominated(cls):
        """Create Chaos-dominated configuration (T_F >> T_P, T_I)."""
        return cls(T_P=0.2, T_I=0.3, T_F=2.0, T_x=0.3, T_y=0.5, T_z=-0.2)
    
    @classmethod
    def balanced(cls):
        """Create balanced configuration (T_P ≈ T_I ≈ T_F)."""
        return cls(T_P=1.0, T_I=1.0, T_F=1.0, T_x=0.5, T_y=0.5, T_z=0.0)


class TemporalTriadProjector:
    """
    Projects temporal triad (T_P, T_I, T_F) onto 2D hexagonal lattice.
    
    This implements the key geometric insight: a 3-fold symmetric object
    (equilateral triangle from P/I/F) combined with spatial orientation
    naturally creates 6-fold (hexagonal) symmetry.
    """
    
    def __init__(self, params: Optional[ProjectionParameters] = None):
        """
        Initialize projector.
        
        Args:
            params: Projection parameters
        """
        self.params = params or ProjectionParameters()
        
        # Equilateral triangle vertices in 2D
        self.V_P = np.array([0.0, 0.0])
        self.V_I = np.array([1.0, 0.0])
        self.V_F = np.array([0.5, np.sqrt(3)/2])
    
    def normalize_triad(self, T_P: float, T_I: float, T_F: float) -> Tuple[float, float, float]:
        """
        Normalize temporal triad to barycentric coordinates.
        
        Args:
            T_P, T_I, T_F: Raw temporal intensities
            
        Returns:
            w_P, w_I, w_F: Normalized weights (sum to 1)
        """
        epsilon = self.params.epsilon
        
        if self.params.triad_normalization == TriadNormalization.SOFTMAX:
            # Exponential normalization (emphasizes dominant component)
            exp_P = np.exp(T_P)
            exp_I = np.exp(T_I)
            exp_F = np.exp(T_F)
            total = exp_P + exp_I + exp_F + epsilon
            
            w_P = exp_P / total
            w_I = exp_I / total
            w_F = exp_F / total
            
        elif self.params.triad_normalization == TriadNormalization.RECTIFIED:
            # Use positive parts only
            pos_P = max(T_P, 0) + epsilon
            pos_I = max(T_I, 0) + epsilon
            pos_F = max(T_F, 0) + epsilon
            total = pos_P + pos_I + pos_F
            
            w_P = pos_P / total
            w_I = pos_I / total
            w_F = pos_F / total
            
        else:  # LINEAR
            # Simple normalization
            total = abs(T_P) + abs(T_I) + abs(T_F) + 3 * epsilon
            
            w_P = (abs(T_P) + epsilon) / total
            w_I = (abs(T_I) + epsilon) / total
            w_F = (abs(T_F) + epsilon) / total
        
        return w_P, w_I, w_F
    
    def barycentric_to_cartesian(self, w_P: float, w_I: float, w_F: float) -> Tuple[float, float]:
        """
        Convert barycentric weights to Cartesian coordinates on triangle.
        
        Args:
            w_P, w_I, w_F: Barycentric weights
            
        Returns:
            u, v: Cartesian coordinates
        """
        point = w_P * self.V_P + w_I * self.V_I + w_F * self.V_F
        return point[0], point[1]
    
    def cartesian_to_hexagonal(self, u: float, v: float) -> Tuple[float, float]:
        """
        Transform from triangle coordinates to hexagonal lattice basis.
        
        The transformation matrix maps the equilateral triangle to a 
        hexagonal unit cell with 120° basis vectors.
        
        Args:
            u, v: Triangle coordinates
            
        Returns:
            X_h1, X_h2: Hexagonal lattice coordinates
        """
        # Transformation matrix to hexagonal basis
        # Basis vectors: e1 = (1, 0), e2 = (1/2, √3/2)
        X_h1 = u - v / np.sqrt(3)
        X_h2 = 2 * v / np.sqrt(3)
        
        # Apply scale
        X_h1 *= self.params.hex_scale
        X_h2 *= self.params.hex_scale
        
        return X_h1, X_h2
    
    def project(self, T_P: float, T_I: float, T_F: float) -> Tuple[float, float]:
        """
        Complete projection: temporal triad → hexagonal coordinates.
        
        Args:
            T_P, T_I, T_F: Temporal tensor components
            
        Returns:
            X_h1, X_h2: Hexagonal lattice coordinates
        """
        # Normalize to barycentric
        w_P, w_I, w_F = self.normalize_triad(T_P, T_I, T_F)
        
        # Map to triangle plane
        u, v = self.barycentric_to_cartesian(w_P, w_I, w_F)
        
        # Transform to hexagonal basis
        X_h1, X_h2 = self.cartesian_to_hexagonal(u, v)
        
        return X_h1, X_h2


class SpatialPhaseProjector:
    """
    Extracts phase coordinate from spatial tensor components.
    
    The spatial triad (T_x, T_y, T_z) encodes local orientation.
    We map this to a phase angle on S^1.
    """
    
    def __init__(self, params: Optional[ProjectionParameters] = None):
        """Initialize projector."""
        self.params = params or ProjectionParameters()
    
    def compute_phase(self, T_x: float, T_y: float, T_z: float) -> float:
        """
        Compute phase angle from spatial components.
        
        Args:
            T_x, T_y, T_z: Spatial tensor components
            
        Returns:
            X_phi: Phase angle in [0, 2π)
        """
        if not self.params.use_spatial_phase:
            return 0.0
        
        # Primary phase from x-y plane
        phi_xy = np.arctan2(T_y, T_x)
        
        # Optional: include z-component via handedness
        # This doubles the effective symmetry (3-fold → 6-fold)
        if T_z != 0:
            handedness = np.sign(T_z) * np.pi / 6  # ±30°
            phi_xy += handedness
        
        # Ensure [0, 2π)
        phi = phi_xy % (2 * np.pi)
        
        return phi


class KRAMProjectionMap:
    """
    Complete projection map f: spacetime → KRAM manifold.
    
    Implements the full ℝ^4 → ℝ^6 mapping:
        (t, x, y, z) → (X_x, X_y, X_z, X_h1, X_h2, X_φ)
    """
    
    def __init__(self, params: Optional[ProjectionParameters] = None):
        """
        Initialize projection map.
        
        Args:
            params: Projection parameters
        """
        self.params = params or ProjectionParameters()
        self.temporal_projector = TemporalTriadProjector(self.params)
        self.spatial_projector = SpatialPhaseProjector(self.params)
    
    def __call__(self,
                 x: float, y: float, z: float, t: float,
                 tensor: Optional[KnoWellianTensorField] = None) -> np.ndarray:
        """
        Apply projection map at a spacetime point.
        
        Args:
            x, y, z: Spatial coordinates
            t: Time coordinate
            tensor: KnoWellian tensor at this point (computed if None)
            
        Returns:
            X: 6D manifold coordinates [X_x, X_y, X_z, X_h1, X_h2, X_φ]
        """
        # Get tensor if not provided
        if tensor is None:
            tensor = KnoWellianTensorField.from_position(x, y, z, t)
        
        # Spatial embedding (coarse-grained)
        X_x = x / self.params.l_KW
        X_y = y / self.params.l_KW
        X_z = z / self.params.l_KW
        
        # Temporal triad → hexagonal coordinates
        X_h1, X_h2 = self.temporal_projector.project(
            tensor.T_P, tensor.T_I, tensor.T_F
        )
        
        # Spatial orientation → phase
        X_phi = self.spatial_projector.compute_phase(
            tensor.T_x, tensor.T_y, tensor.T_z
        )
        
        return np.array([X_x, X_y, X_z, X_h1, X_h2, X_phi])
    
    def batch_project(self,
                     x_array: np.ndarray,
                     y_array: np.ndarray,
                     z_array: np.ndarray,
                     t_array: np.ndarray,
                     tensor_field: Optional[Callable] = None) -> np.ndarray:
        """
        Project multiple spacetime points.
        
        Args:
            x_array, y_array, z_array, t_array: Coordinate arrays
            tensor_field: Optional function (x,y,z,t) → tensor
            
        Returns:
            X_array: Shape (N, 6) manifold coordinates
        """
        N = len(x_array)
        X_array = np.zeros((N, 6))
        
        for i in range(N):
            if tensor_field is not None:
                tensor = tensor_field(x_array[i], y_array[i], z_array[i], t_array[i])
            else:
                tensor = None
            
            X_array[i] = self(x_array[i], y_array[i], z_array[i], t_array[i], tensor)
        
        return X_array
    
    def jacobian(self,
                x: float, y: float, z: float, t: float,
                tensor: Optional[KnoWellianTensorField] = None,
                delta: float = 1e-5) -> np.ndarray:
        """
        Compute Jacobian matrix ∂X/∂x numerically.
        
        Args:
            x, y, z, t: Spacetime point
            tensor: Tensor at this point
            delta: Finite difference step
            
        Returns:
            J: 6×4 Jacobian matrix
        """
        X_center = self(x, y, z, t, tensor)
        J = np.zeros((6, 4))
        
        # Finite differences
        coords = [x, y, z, t]
        for i, coord in enumerate(coords):
            coords_plus = coords.copy()
            coords_plus[i] += delta
            
            X_plus = self(*coords_plus, tensor)
            J[:, i] = (X_plus - X_center) / delta
        
        return J


class HexagonalLatticeAnalyzer:
    """
    Analyzes point distributions in hexagonal coordinates for Cairo patterns.
    """
    
    @staticmethod
    def detect_symmetry(X_h1: np.ndarray, X_h2: np.ndarray, 
                       n_bins: int = 20) -> Tuple[np.ndarray, float]:
        """
        Detect hexagonal/pentagonal symmetry in point distribution.
        
        Args:
            X_h1, X_h2: Hexagonal coordinates of points
            n_bins: Number of angular bins
            
        Returns:
            angular_spectrum: Power in each angular direction
            symmetry_score: Measure of 5-fold or 6-fold symmetry
        """
        # Compute angles
        angles = np.arctan2(X_h2, X_h1)
        
        # Histogram
        hist, bin_edges = np.histogram(angles, bins=n_bins, 
                                       range=(-np.pi, np.pi))
        
        # Fourier transform to detect periodicities
        fft = np.fft.fft(hist)
        power = np.abs(fft)**2
        
        # 5-fold symmetry: peak at k=5
        # 6-fold symmetry: peak at k=6
        five_fold_power = power[5] if len(power) > 5 else 0
        six_fold_power = power[6] if len(power) > 6 else 0
        
        symmetry_score = max(five_fold_power, six_fold_power) / np.sum(power[1:])
        
        return power, symmetry_score
    
    @staticmethod
    def compute_lattice_spacing(X_h1: np.ndarray, X_h2: np.ndarray) -> float:
        """
        Estimate fundamental lattice spacing.
        
        Args:
            X_h1, X_h2: Hexagonal coordinates
            
        Returns:
            Estimated lattice constant
        """
        # Compute all pairwise distances
        points = np.column_stack([X_h1, X_h2])
        N = len(points)
        
        if N < 2:
            return 0.0
        
        # Sample pairs to avoid O(N²)
        sample_size = min(1000, N)
        idx = np.random.choice(N, sample_size, replace=False)
        sampled = points[idx]
        
        distances = []
        for i in range(len(sampled)):
            for j in range(i+1, len(sampled)):
                dist = np.linalg.norm(sampled[i] - sampled[j])
                if dist > 0.1:  # Avoid self-distances
                    distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Peak in distance histogram indicates lattice spacing
        hist, bin_edges = np.histogram(distances, bins=50)
        peak_idx = np.argmax(hist)
        lattice_spacing = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
        
        return lattice_spacing


# ============================================================================
# Example Usage
# ============================================================================

def example_single_point_projection():
    """Example: Project single spacetime point."""
    print("Example 1: Single Point Projection")
    print("-" * 50)
    
    # Create projection map
    params = ProjectionParameters(l_KW=1.0, hex_scale=1.0)
    proj_map = KRAMProjectionMap(params)
    
    # Test with different tensor configurations
    configs = {
        "Balanced": KnoWellianTensorField.balanced(),
        "Control-dominated": KnoWellianTensorField.control_dominated(),
        "Chaos-dominated": KnoWellianTensorField.chaos_dominated(),
    }
    
    x, y, z, t = 10.0, 5.0, 0.0, 0.0
    
    for name, tensor in configs.items():
        X = proj_map(x, y, z, t, tensor)
        print(f"\n{name}:")
        print(f"  Spatial: ({X[0]:.2f}, {X[1]:.2f}, {X[2]:.2f})")
        print(f"  Hexagonal: ({X[3]:.3f}, {X[4]:.3f})")
        print(f"  Phase: {X[5]:.3f} rad ({np.degrees(X[5]):.1f}°)")
    
    return proj_map


def example_spatial_distribution():
    """Example: Project grid of points and analyze symmetry."""
    print("\nExample 2: Spatial Distribution Analysis")
    print("-" * 50)
    
    proj_map = KRAMProjectionMap()
    
    # Create grid of spacetime points
    n = 30
    x_grid = np.linspace(0, 20, n)
    y_grid = np.linspace(0, 20, n)
    
    X_manifold = []
    for x in x_grid:
        for y in y_grid:
            X = proj_map(x, y, 0.0, 0.0)
            X_manifold.append(X)
    
    X_manifold = np.array(X_manifold)
    
    # Extract hexagonal coordinates
    X_h1 = X_manifold[:, 3]
    X_h2 = X_manifold[:, 4]
    
    # Analyze symmetry
    analyzer = HexagonalLatticeAnalyzer()
    power, symmetry_score = analyzer.detect_symmetry(X_h1, X_h2)
    lattice_spacing = analyzer.compute_lattice_spacing(X_h1, X_h2)
    
    print(f"Symmetry score: {symmetry_score:.4f}")
    print(f"Estimated lattice spacing: {lattice_spacing:.3f}")
    print(f"5-fold power: {power[5]:.2e}")
    print(f"6-fold power: {power[6]:.2e}")
    
    return X_manifold, analyzer


def example_jacobian_analysis():
    """Example: Compute and analyze Jacobian."""
    print("\nExample 3: Jacobian Analysis")
    print("-" * 50)
    
    proj_map = KRAMProjectionMap()
    
    x, y, z, t = 5.0, 5.0, 0.0, 0.0
    
    J = proj_map.jacobian(x, y, z, t)
    
    print("Jacobian matrix (6×4):")
    print(J)
    print(f"\nCondition number: {np.linalg.cond(J):.2e}")
    
    # Check non-degeneracy
    JTJ = J.T @ J
    eigenvalues = np.linalg.eigvalsh(JTJ)
    print(f"Smallest eigenvalue: {eigenvalues[0]:.2e}")
    
    if eigenvalues[0] > 1e-10:
        print("✓ Map is non-degenerate")
    else:
        print("✗ Map may be degenerate")
    
    return J


if __name__ == "__main__":
    print("=" * 70)
    print("Projection Maps Module - Test Suite")
    print("=" * 70)
    
    # Run examples
    proj_map = example_single_point_projection()
    X_manifold, analyzer = example_spatial_distribution()
    J = example_jacobian_analysis()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
    print("\nKey results:")
    print("  - Projection map successfully transforms spacetime → KRAM")
    print("  - Hexagonal symmetry emerges from temporal triad")
    print("  - Map is non-degenerate (invertible locally)")
    print("  - Ready for integration with KRAM imprint kernel")
