"""
Control-Chaos Forcing Module
============================

Implements the Control and Chaos field generators for KnoWellian Universe Theory.

Control Field (φ_C): Coherent, deterministic forcing from the Past (t_P)
    - Time-coherent oscillations
    - Spatially structured patterns
    - Represents established order and particle-like emergence

Chaos Field (φ_X): Incoherent, stochastic forcing from the Future (t_F)
    - Rapidly decorrelating noise
    - Wave-like potential
    - Represents entropic dissolution and novelty

The balance between these fields determines whether the system exhibits:
    - Sharp resonances (Control-dominated)
    - Broad acoustic-like peaks (balanced)
    - Diffuse spectra (Chaos-dominated)

Author: David Noel Lynch
Date: 2025
License: MIT
"""

import numpy as np
from scipy import ndimage, signal
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum


class ForceType(Enum):
    """Types of forcing patterns."""
    CONTROL = "control"
    CHAOS = "chaos"
    BALANCED = "balanced"


@dataclass
class ControlParameters:
    """
    Parameters for Control field (coherent forcing from Past).
    
    Attributes:
        amplitude: Overall strength of Control forcing
        omega: Angular frequency of coherent oscillation
        k_modes: List of preferred spatial wavenumbers to excite
        phase: Global phase offset
        spatial_coherence: Length scale of spatial correlations
        temporal_coherence: Time scale over which patterns persist
    """
    amplitude: float = 1.0
    omega: float = 0.5
    k_modes: List[float] = None
    phase: float = 0.0
    spatial_coherence: float = 5.0
    temporal_coherence: float = np.inf  # Fully coherent by default
    
    def __post_init__(self):
        if self.k_modes is None:
            self.k_modes = [0.5]  # Default single mode


@dataclass
class ChaosParameters:
    """
    Parameters for Chaos field (incoherent forcing from Future).
    
    Attributes:
        amplitude: Overall strength of Chaos forcing
        spatial_correlation: Length scale of spatial noise correlation
        temporal_correlation: Time scale of temporal decorrelation
        spectrum_power: Power-law exponent for noise spectrum (0=white, 2=Brownian)
        refresh_rate: How often to generate new noise field (in time units)
    """
    amplitude: float = 1.0
    spatial_correlation: float = 2.0
    temporal_correlation: float = 1.0
    spectrum_power: float = 1.0  # Pink noise (1/f spectrum)
    refresh_rate: float = 0.1


class ControlField:
    """
    Generator for Control field - coherent, deterministic forcing.
    
    Represents the outward flow from Ultimaton (Past realm), creating
    deterministic, particle-like structures through resonant pumping.
    """
    
    def __init__(self, 
                 grid_shape: Tuple[int, ...],
                 params: Optional[ControlParameters] = None,
                 dx: float = 1.0):
        """
        Initialize Control field generator.
        
        Args:
            grid_shape: Shape of spatial grid
            params: Control parameters
            dx: Grid spacing
        """
        self.grid_shape = grid_shape
        self.ndim = len(grid_shape)
        self.params = params or ControlParameters()
        self.dx = dx
        
        # Create spatial mode patterns
        self._generate_spatial_modes()
        
    def _generate_spatial_modes(self):
        """
        Generate spatial mode patterns for each k-mode.
        
        Creates standing wave patterns at specified wavenumbers.
        """
        self.mode_patterns = []
        
        # Create coordinate grids
        coords = [np.arange(N) * self.dx for N in self.grid_shape]
        grids = np.meshgrid(*coords, indexing='ij')
        
        for k_mode in self.params.k_modes:
            if self.ndim == 1:
                # 1D standing wave
                pattern = np.cos(k_mode * grids[0])
            elif self.ndim == 2:
                # 2D radial mode
                x, y = grids
                r = np.sqrt(x**2 + y**2)
                pattern = np.cos(k_mode * r)
            else:
                # 3D spherical mode
                r = np.sqrt(sum(g**2 for g in grids))
                pattern = np.cos(k_mode * r)
            
            # Normalize
            pattern = pattern / np.max(np.abs(pattern))
            self.mode_patterns.append(pattern)
    
    def generate(self, t: float) -> np.ndarray:
        """
        Generate Control field at time t.
        
        Args:
            t: Current time
            
        Returns:
            Control field φ_C(x, t)
        """
        # Time-coherent oscillation
        time_factor = np.cos(self.params.omega * t + self.params.phase)
        
        # Sum over spatial modes
        field = np.zeros(self.grid_shape)
        for pattern in self.mode_patterns:
            field += pattern
        
        # Normalize by number of modes
        field = field / len(self.mode_patterns)
        
        # Apply amplitude and time modulation
        field = self.params.amplitude * time_factor * field
        
        # Add spatial coherence envelope if specified
        if self.params.spatial_coherence < np.inf:
            envelope = self._spatial_envelope()
            field = field * envelope
        
        return field
    
    def _spatial_envelope(self) -> np.ndarray:
        """
        Create Gaussian spatial envelope for localized forcing.
        
        Returns:
            Spatial envelope function
        """
        coords = [np.arange(N) * self.dx for N in self.grid_shape]
        grids = np.meshgrid(*coords, indexing='ij')
        
        # Center coordinates
        centers = [N * self.dx / 2 for N in self.grid_shape]
        
        # Compute distance from center
        r_squared = sum((g - c)**2 for g, c in zip(grids, centers))
        
        # Gaussian envelope
        return np.exp(-r_squared / (2 * self.params.spatial_coherence**2))
    
    def add_harmonic(self, k_mode: float, amplitude_factor: float = 1.0):
        """
        Add additional harmonic mode to Control field.
        
        Args:
            k_mode: Wavenumber of new mode
            amplitude_factor: Relative amplitude
        """
        self.params.k_modes.append(k_mode)
        self._generate_spatial_modes()


class ChaosField:
    """
    Generator for Chaos field - incoherent, stochastic forcing.
    
    Represents the inward collapse from Entropium (Future realm), creating
    wave-like potentiality and entropic dissolution.
    """
    
    def __init__(self,
                 grid_shape: Tuple[int, ...],
                 params: Optional[ChaosParameters] = None,
                 dx: float = 1.0):
        """
        Initialize Chaos field generator.
        
        Args:
            grid_shape: Shape of spatial grid
            params: Chaos parameters
            dx: Grid spacing
        """
        self.grid_shape = grid_shape
        self.ndim = len(grid_shape)
        self.params = params or ChaosParameters()
        self.dx = dx
        
        # Cache for temporal correlation
        self._noise_cache = None
        self._last_refresh_time = -np.inf
        
        # Setup colored noise generator
        self._setup_noise_spectrum()
    
    def _setup_noise_spectrum(self):
        """
        Setup power spectrum for colored noise generation.
        
        Creates 1/f^β spectrum in Fourier space.
        """
        # Create wavenumber grids
        k_grids = []
        for N in self.grid_shape:
            k = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
            k_grids.append(k)
        
        k_arrays = np.meshgrid(*k_grids, indexing='ij')
        self.k_radial = np.sqrt(sum(k**2 for k in k_arrays))
        
        # Avoid division by zero at k=0
        self.k_radial[self.k_radial == 0] = 1e-10
        
        # Power spectrum: P(k) ∝ 1/k^β
        beta = self.params.spectrum_power
        self.noise_spectrum = 1.0 / (self.k_radial ** beta)
        
        # Normalize
        self.noise_spectrum = self.noise_spectrum / np.mean(self.noise_spectrum)
    
    def _generate_colored_noise(self) -> np.ndarray:
        """
        Generate spatially-correlated colored noise with specified spectrum.
        
        Returns:
            Noise field with desired power spectrum
        """
        # White noise in Fourier space
        noise_fft = (np.random.randn(*self.grid_shape) + 
                    1j * np.random.randn(*self.grid_shape))
        
        # Apply power spectrum
        noise_fft = noise_fft * np.sqrt(self.noise_spectrum)
        
        # Transform to real space
        noise = np.real(np.fft.ifftn(noise_fft))
        
        # Additional spatial smoothing
        if self.params.spatial_correlation > 0:
            sigma = self.params.spatial_correlation / self.dx
            noise = ndimage.gaussian_filter(noise, sigma=sigma)
        
        # Normalize to unit variance
        noise = noise / np.std(noise)
        
        return noise
    
    def generate(self, t: float) -> np.ndarray:
        """
        Generate Chaos field at time t.
        
        Args:
            t: Current time
            
        Returns:
            Chaos field φ_X(x, t)
        """
        # Check if we need to refresh the noise field
        time_since_refresh = t - self._last_refresh_time
        
        if (self._noise_cache is None or 
            time_since_refresh >= self.params.refresh_rate):
            # Generate new noise field
            self._noise_cache = self._generate_colored_noise()
            self._last_refresh_time = t
        
        # Apply temporal decorrelation if specified
        if self.params.temporal_correlation < np.inf:
            # Exponential decay factor
            decay = np.exp(-time_since_refresh / self.params.temporal_correlation)
            
            # Mix cached noise with new noise
            if decay < 0.9:  # Refresh when decorrelated
                new_noise = self._generate_colored_noise()
                field = decay * self._noise_cache + np.sqrt(1 - decay**2) * new_noise
                self._noise_cache = field
            else:
                field = self._noise_cache
        else:
            field = self._noise_cache
        
        # Apply amplitude
        return self.params.amplitude * field
    
    def reset(self):
        """Reset temporal correlation cache."""
        self._noise_cache = None
        self._last_refresh_time = -np.inf


class ControlChaosForcing:
    """
    Combined Control-Chaos forcing generator.
    
    Implements the fundamental KUT dynamic: the balance between
    deterministic order (Control) and stochastic potential (Chaos).
    """
    
    def __init__(self,
                 grid_shape: Tuple[int, ...],
                 control_params: Optional[ControlParameters] = None,
                 chaos_params: Optional[ChaosParameters] = None,
                 dx: float = 1.0):
        """
        Initialize combined forcing generator.
        
        Args:
            grid_shape: Shape of spatial grid
            control_params: Control field parameters
            chaos_params: Chaos field parameters
            dx: Grid spacing
        """
        self.grid_shape = grid_shape
        self.dx = dx
        
        # Initialize component generators
        self.control = ControlField(grid_shape, control_params, dx)
        self.chaos = ChaosField(grid_shape, chaos_params, dx)
        
        # Balance parameter
        self.control_fraction = 0.5  # Equal by default
    
    def generate(self, t: float, force_type: ForceType = ForceType.BALANCED) -> np.ndarray:
        """
        Generate combined forcing field at time t.
        
        Args:
            t: Current time
            force_type: Type of forcing (CONTROL, CHAOS, or BALANCED)
            
        Returns:
            Combined forcing field F_CC(x, t)
        """
        if force_type == ForceType.CONTROL:
            return self.control.generate(t)
        elif force_type == ForceType.CHAOS:
            return self.chaos.generate(t)
        else:  # BALANCED
            f_control = self.control.generate(t)
            f_chaos = self.chaos.generate(t)
            
            return (self.control_fraction * f_control + 
                   (1 - self.control_fraction) * f_chaos)
    
    def set_balance(self, control_fraction: float):
        """
        Set the Control-Chaos balance.
        
        Args:
            control_fraction: Fraction of Control (0=pure Chaos, 1=pure Control)
        """
        self.control_fraction = np.clip(control_fraction, 0.0, 1.0)
    
    def sweep_balance(self, 
                     t: float,
                     n_samples: int = 10) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generate forcing fields for range of Control-Chaos balances.
        
        Args:
            t: Time point
            n_samples: Number of balance points to sample
            
        Returns:
            balance_values: Array of control fractions
            fields: List of forcing fields
        """
        balance_values = np.linspace(0, 1, n_samples)
        fields = []
        
        for balance in balance_values:
            self.set_balance(balance)
            fields.append(self.generate(t))
        
        return balance_values, fields


class VacuumStructure:
    """
    Generator for background vacuum structure (Cairo Q-Lattice approximation).
    
    Provides the geometric substrate that biases KRAM evolution.
    """
    
    def __init__(self,
                 grid_shape: Tuple[int, ...],
                 wavelength: float = 10.0,
                 amplitude: float = 0.3,
                 dx: float = 1.0,
                 lattice_type: str = 'hexagonal'):
        """
        Initialize vacuum structure generator.
        
        Args:
            grid_shape: Shape of spatial grid
            wavelength: Fundamental wavelength of lattice
            amplitude: Pattern amplitude
            dx: Grid spacing
            lattice_type: Type of lattice ('hexagonal', 'pentagonal', 'square')
        """
        self.grid_shape = grid_shape
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.dx = dx
        self.lattice_type = lattice_type
        
        # Generate static pattern
        self.pattern = self._generate_pattern()
    
    def _generate_pattern(self) -> np.ndarray:
        """
        Generate spatial lattice pattern.
        
        Returns:
            Lattice pattern field
        """
        if self.lattice_type == 'hexagonal':
            return self._hexagonal_pattern()
        elif self.lattice_type == 'pentagonal':
            return self._pentagonal_pattern()
        elif self.lattice_type == 'square':
            return self._square_pattern()
        else:
            raise ValueError(f"Unknown lattice type: {self.lattice_type}")
    
    def _hexagonal_pattern(self) -> np.ndarray:
        """Create hexagonal lattice (6-fold symmetry)."""
        if len(self.grid_shape) != 2:
            raise ValueError("Hexagonal pattern only for 2D grids")
        
        x = np.arange(self.grid_shape[0]) * self.dx
        y = np.arange(self.grid_shape[1]) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        k = 2 * np.pi / self.wavelength
        
        # Three plane waves at 120° create hexagonal pattern
        pattern = (np.cos(k * X) + 
                  np.cos(k * (0.5 * X - np.sqrt(3)/2 * Y)) +
                  np.cos(k * (0.5 * X + np.sqrt(3)/2 * Y)))
        
        return self.amplitude * pattern / 3.0
    
    def _pentagonal_pattern(self) -> np.ndarray:
        """
        Create pentagonal lattice approximation (5-fold symmetry).
        
        True Cairo pentagonal tiling is aperiodic, but we approximate
        with five plane waves at 72° angles.
        """
        if len(self.grid_shape) != 2:
            raise ValueError("Pentagonal pattern only for 2D grids")
        
        x = np.arange(self.grid_shape[0]) * self.dx
        y = np.arange(self.grid_shape[1]) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        k = 2 * np.pi / self.wavelength
        
        # Five plane waves at 72° create pentagonal approximation
        pattern = np.zeros_like(X)
        for i in range(5):
            angle = 2 * np.pi * i / 5
            kx = k * np.cos(angle)
            ky = k * np.sin(angle)
            pattern += np.cos(kx * X + ky * Y)
        
        return self.amplitude * pattern / 5.0
    
    def _square_pattern(self) -> np.ndarray:
        """Create square lattice (4-fold symmetry)."""
        if len(self.grid_shape) != 2:
            raise ValueError("Square pattern only for 2D grids")
        
        x = np.arange(self.grid_shape[0]) * self.dx
        y = np.arange(self.grid_shape[1]) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        k = 2 * np.pi / self.wavelength
        
        pattern = np.cos(k * X) + np.cos(k * Y)
        
        return self.amplitude * pattern / 2.0
    
    def get_pattern(self) -> np.ndarray:
        """Return the static vacuum pattern."""
        return self.pattern.copy()


# ============================================================================
# Complete Forcing Scenarios
# ============================================================================

def create_cmb_forcing(grid_shape: Tuple[int, int],
                      control_strength: float = 1.5,
                      chaos_strength: float = 1.2,
                      pump_frequency: float = 0.5,
                      vacuum_wavelength: float = 8.0,
                      dx: float = 1.0) -> Tuple[ControlChaosForcing, VacuumStructure]:
    """
    Create forcing configuration for CMB-like spectrum generation.
    
    Args:
        grid_shape: 2D grid shape
        control_strength: Amplitude of coherent Control pump
        chaos_strength: Amplitude of incoherent Chaos
        pump_frequency: Oscillation frequency of Control pump
        vacuum_wavelength: Wavelength of vacuum lattice
        dx: Grid spacing
        
    Returns:
        forcing: ControlChaosForcing generator
        vacuum: VacuumStructure with lattice pattern
    """
    # Control parameters: coherent pump
    control_params = ControlParameters(
        amplitude=control_strength,
        omega=pump_frequency,
        k_modes=[2 * np.pi / vacuum_wavelength],  # Resonant with vacuum
        spatial_coherence=np.inf  # Global coherence
    )
    
    # Chaos parameters: rapidly decorrelating noise
    chaos_params = ChaosParameters(
        amplitude=chaos_strength,
        spatial_correlation=2.0,
        temporal_correlation=1.0,
        spectrum_power=1.0,  # Pink noise
        refresh_rate=0.1
    )
    
    # Combined forcing
    forcing = ControlChaosForcing(grid_shape, control_params, chaos_params, dx)
    
    # Vacuum structure (hexagonal approximation to Cairo)
    vacuum = VacuumStructure(
        grid_shape=grid_shape,
        wavelength=vacuum_wavelength,
        amplitude=0.3,
        dx=dx,
        lattice_type='hexagonal'
    )
    
    return forcing, vacuum


def create_particle_imprint(grid_shape: Tuple[int, ...],
                           center: Optional[Tuple[float, ...]] = None,
                           amplitude: float = 2.0,
                           width: float = 3.0,
                           dx: float = 1.0) -> np.ndarray:
    """
    Create localized particle imprint (KnoWellian Soliton).
    
    Args:
        grid_shape: Shape of grid
        center: Center coordinates (defaults to grid center)
        amplitude: Peak amplitude
        width: Gaussian width
        dx: Grid spacing
        
    Returns:
        Gaussian imprint field
    """
    if center is None:
        center = tuple(N / 2 for N in grid_shape)
    
    coords = [np.arange(N) * dx for N in grid_shape]
    grids = np.meshgrid(*coords, indexing='ij')
    
    r_squared = sum((g - c * dx)**2 for g, c in zip(grids, center))
    
    return amplitude * np.exp(-r_squared / (2 * width**2))


# ============================================================================
# Example Usage
# ============================================================================

def example_control_only():
    """Example: Pure Control forcing (sharp resonances)."""
    print("Example 1: Pure Control Field")
    print("-" * 50)
    
    control_params = ControlParameters(
        amplitude=1.0,
        omega=0.5,
        k_modes=[0.5, 1.0],  # Two harmonics
    )
    
    control = ControlField(grid_shape=(64, 64), params=control_params)
    
    # Generate at different times
    times = [0, np.pi/2, np.pi]
    for t in times:
        field = control.generate(t)
        print(f"t={t:.2f}: mean={np.mean(field):.3f}, std={np.std(field):.3f}")
    
    return control


def example_chaos_only():
    """Example: Pure Chaos forcing (stochastic)."""
    print("\nExample 2: Pure Chaos Field")
    print("-" * 50)
    
    chaos_params = ChaosParameters(
        amplitude=1.0,
        spatial_correlation=2.0,
        temporal_correlation=5.0,
        spectrum_power=1.0
    )
    
    chaos = ChaosField(grid_shape=(64, 64), params=chaos_params)
    
    # Generate at different times
    times = [0, 5, 10]
    for t in times:
        field = chaos.generate(t)
        print(f"t={t:.2f}: mean={np.mean(field):.3f}, std={np.std(field):.3f}")
    
    return chaos


def example_balanced_forcing():
    """Example: Balanced Control-Chaos (CMB-like)."""
    print("\nExample 3: Balanced Control-Chaos")
    print("-" * 50)
    
    forcing, vacuum = create_cmb_forcing(
        grid_shape=(64, 64),
        control_strength=1.5,
        chaos_strength=1.2,
        pump_frequency=0.5,
        vacuum_wavelength=8.0
    )
    
    # Generate total forcing at t=10
    t = 10.0
    f_control = forcing.control.generate(t)
    f_chaos = forcing.chaos.generate(t)
    f_vacuum = vacuum.get_pattern()
    f_total = forcing.generate(t) + f_vacuum
    
    print(f"Control field: mean={np.mean(f_control):.3f}, std={np.std(f_control):.3f}")
    print(f"Chaos field: mean={np.mean(f_chaos):.3f}, std={np.std(f_chaos):.3f}")
    print(f"Vacuum pattern: mean={np.mean(f_vacuum):.3f}, std={np.std(f_vacuum):.3f}")
    print(f"Total forcing: mean={np.mean(f_total):.3f}, std={np.std(f_total):.3f}")
    
    return forcing, vacuum


if __name__ == "__main__":
    print("=" * 70)
    print("Control-Chaos Forcing Module - Test Suite")
    print("=" * 70)
    
    # Run examples
    control = example_control_only()
    chaos = example_chaos_only()
    forcing, vacuum = example_balanced_forcing()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
