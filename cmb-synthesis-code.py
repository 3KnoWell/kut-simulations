"""
CMB Synthesis Module
===================

Implements projection of KRAM source power spectrum to CMB angular power spectrum.

The fundamental relation:
    C_ℓ = (2/π) ∫ dk k² P_S(k) |Δ_ℓ(k)|²

where:
    P_S(k): Source power spectrum from KRAM evolution
    Δ_ℓ(k): Radiation transfer function (spherical Bessel projection)
    C_ℓ: Angular power spectrum (observable in CMB)

This module bridges the gap between KRAM field dynamics and cosmological observables,
allowing direct comparison with Planck satellite measurements.

Author: David Noel Lynch
Date: 2025
License: MIT
"""

import numpy as np
from scipy import special, integrate, interpolate
from typing import Tuple, Optional, Callable, Dict
from dataclasses import dataclass
import warnings


@dataclass
class CosmologicalParameters:
    """
    Standard cosmological parameters for CMB calculations.
    
    Attributes:
        chi_star: Comoving distance to last scattering surface (Mpc)
        delta_chi: Thickness of last scattering shell (Mpc)
        H0: Hubble constant (km/s/Mpc)
        Omega_m: Matter density parameter
        Omega_Lambda: Dark energy density parameter
        T_CMB: CMB temperature today (K)
    """
    chi_star: float = 14000.0  # Mpc
    delta_chi: float = 100.0    # Mpc
    H0: float = 67.4           # km/s/Mpc
    Omega_m: float = 0.315
    Omega_Lambda: float = 0.685
    T_CMB: float = 2.725       # Kelvin


class SphericalBesselCache:
    """
    Efficient caching and computation of spherical Bessel functions.
    
    j_ℓ(x) = √(π/2x) J_{ℓ+1/2}(x)
    
    These are expensive to compute repeatedly, so we cache values.
    """
    
    def __init__(self, ell_max: int = 2000, x_max: float = 5000.0, n_samples: int = 1000):
        """
        Initialize Bessel function cache.
        
        Args:
            ell_max: Maximum multipole to cache
            x_max: Maximum argument value
            n_samples: Number of x points to sample
        """
        self.ell_max = ell_max
        self.x_max = x_max
        self.n_samples = n_samples
        
        # Create cache grid
        self.x_grid = np.linspace(0.01, x_max, n_samples)
        self.ell_grid = np.arange(2, ell_max + 1)
        
        # Precompute on grid
        self._cache = {}
        print("Precomputing spherical Bessel functions...")
        for ell in self.ell_grid:
            if ell % 100 == 0:
                print(f"  ℓ = {ell}")
            self._cache[ell] = self._compute_jell(ell, self.x_grid)
    
    def _compute_jell(self, ell: int, x: np.ndarray) -> np.ndarray:
        """
        Compute spherical Bessel function j_ℓ(x).
        
        Args:
            ell: Multipole order
            x: Argument values
            
        Returns:
            j_ℓ(x)
        """
        # Handle x=0 case
        x = np.atleast_1d(x)
        result = np.zeros_like(x)
        
        # For small x, use series expansion
        small_x = x < 0.1
        if np.any(small_x):
            x_small = x[small_x]
            # j_ℓ(x) ≈ x^ℓ / (2ℓ+1)!! for small x
            factorial_term = special.factorial2(2*ell + 1)
            result[small_x] = x_small**ell / factorial_term
        
        # For normal x, use standard formula
        normal_x = ~small_x
        if np.any(normal_x):
            x_normal = x[normal_x]
            # j_ℓ(x) = √(π/2x) J_{ℓ+1/2}(x)
            result[normal_x] = np.sqrt(np.pi / (2 * x_normal)) * special.jv(ell + 0.5, x_normal)
        
        return result
    
    def get(self, ell: int, x: np.ndarray) -> np.ndarray:
        """
        Get spherical Bessel function values (cached or interpolated).
        
        Args:
            ell: Multipole order
            x: Argument values
            
        Returns:
            j_ℓ(x)
        """
        if ell not in self._cache:
            # Compute on the fly for uncached ℓ
            return self._compute_jell(ell, x)
        
        # Interpolate from cache
        cached_values = self._cache[ell]
        interpolator = interpolate.interp1d(
            self.x_grid, 
            cached_values,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        
        return interpolator(x)


class VisibilityFunction:
    """
    Represents the visibility function W(χ) for CMB photons.
    
    W(χ) describes the probability distribution of last scattering as 
    a function of comoving distance. For simplicity, we use a Gaussian.
    """
    
    def __init__(self, cosmo_params: Optional[CosmologicalParameters] = None):
        """
        Initialize visibility function.
        
        Args:
            cosmo_params: Cosmological parameters
        """
        self.params = cosmo_params or CosmologicalParameters()
    
    def __call__(self, chi: np.ndarray) -> np.ndarray:
        """
        Evaluate visibility function at comoving distance χ.
        
        Args:
            chi: Comoving distance (Mpc)
            
        Returns:
            W(χ)
        """
        # Gaussian centered at chi_star with width delta_chi
        exp_term = -((chi - self.params.chi_star)**2 / 
                     (2 * self.params.delta_chi**2))
        
        # Normalize
        norm = 1.0 / (np.sqrt(2 * np.pi) * self.params.delta_chi)
        
        return norm * np.exp(exp_term)
    
    def thin_shell_approximation(self) -> float:
        """
        Return chi_star for thin shell approximation: W(χ) ≈ δ(χ - χ_star).
        
        Returns:
            chi_star (Mpc)
        """
        return self.params.chi_star


class TransferFunction:
    """
    Radiation transfer function Δ_ℓ(k).
    
    For thin shell approximation:
        Δ_ℓ(k) ≈ j_ℓ(k χ_star)
    
    For thick shell:
        Δ_ℓ(k) = ∫ dχ W(χ) j_ℓ(k χ)
    """
    
    def __init__(self,
                 visibility: VisibilityFunction,
                 bessel_cache: Optional[SphericalBesselCache] = None,
                 thin_shell: bool = False):
        """
        Initialize transfer function.
        
        Args:
            visibility: Visibility function W(χ)
            bessel_cache: Cached spherical Bessel functions
            thin_shell: Use thin shell approximation
        """
        self.visibility = visibility
        self.bessel_cache = bessel_cache or SphericalBesselCache()
        self.thin_shell = thin_shell
    
    def compute(self, ell: int, k: np.ndarray) -> np.ndarray:
        """
        Compute transfer function Δ_ℓ(k).
        
        Args:
            ell: Multipole order
            k: Wavenumber array (1/Mpc)
            
        Returns:
            Δ_ℓ(k)
        """
        if self.thin_shell:
            return self._thin_shell_transfer(ell, k)
        else:
            return self._thick_shell_transfer(ell, k)
    
    def _thin_shell_transfer(self, ell: int, k: np.ndarray) -> np.ndarray:
        """
        Compute transfer function using thin shell approximation.
        
        Args:
            ell: Multipole order
            k: Wavenumber array
            
        Returns:
            j_ℓ(k χ_star)
        """
        chi_star = self.visibility.thin_shell_approximation()
        x = k * chi_star
        return self.bessel_cache.get(ell, x)
    
    def _thick_shell_transfer(self, ell: int, k: np.ndarray) -> np.ndarray:
        """
        Compute transfer function with full radial integration.
        
        Args:
            ell: Multipole order
            k: Wavenumber array
            
        Returns:
            ∫ dχ W(χ) j_ℓ(k χ)
        """
        # Integration range
        chi_min = self.visibility.params.chi_star - 5 * self.visibility.params.delta_chi
        chi_max = self.visibility.params.chi_star + 5 * self.visibility.params.delta_chi
        chi_min = max(chi_min, 1.0)  # Avoid chi=0
        
        chi_grid = np.linspace(chi_min, chi_max, 100)
        W_chi = self.visibility(chi_grid)
        
        # Integrate for each k
        result = np.zeros_like(k)
        for i, k_val in enumerate(k):
            x = k_val * chi_grid
            j_ell = self.bessel_cache.get(ell, x)
            integrand = W_chi * j_ell
            result[i] = np.trapz(integrand, chi_grid)
        
        return result


class CMBSynthesizer:
    """
    Main class for synthesizing CMB angular power spectrum from KRAM source.
    """
    
    def __init__(self,
                 cosmo_params: Optional[CosmologicalParameters] = None,
                 thin_shell: bool = True,
                 ell_max: int = 2000):
        """
        Initialize CMB synthesizer.
        
        Args:
            cosmo_params: Cosmological parameters
            thin_shell: Use thin shell approximation (faster)
            ell_max: Maximum multipole to compute
        """
        self.params = cosmo_params or CosmologicalParameters()
        self.ell_max = ell_max
        
        # Setup components
        print("Initializing CMB Synthesizer...")
        self.bessel_cache = SphericalBesselCache(ell_max=ell_max)
        self.visibility = VisibilityFunction(self.params)
        self.transfer = TransferFunction(self.visibility, self.bessel_cache, thin_shell)
        
        print("Ready for synthesis.")
    
    def source_to_angular_spectrum(self,
                                   k_values: np.ndarray,
                                   P_S_k: np.ndarray,
                                   ell_values: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project source power spectrum P_S(k) to angular spectrum C_ℓ.
        
        Implements:
            C_ℓ = (2/π) ∫ dk k² P_S(k) |Δ_ℓ(k)|²
        
        Args:
            k_values: Source wavenumbers (1/Mpc)
            P_S_k: Source power spectrum P_S(k)
            ell_values: Multipoles to compute (defaults to 2 to ell_max)
            
        Returns:
            ell_values: Multipole array
            C_ell: Angular power spectrum
        """
        if ell_values is None:
            ell_values = np.arange(2, self.ell_max + 1)
        
        # Ensure k is sorted and positive
        k_values = np.array(k_values)
        P_S_k = np.array(P_S_k)
        
        # Remove k=0 if present
        nonzero = k_values > 0
        k_values = k_values[nonzero]
        P_S_k = P_S_k[nonzero]
        
        # Sort
        sort_idx = np.argsort(k_values)
        k_values = k_values[sort_idx]
        P_S_k = P_S_k[sort_idx]
        
        # Interpolate P_S for smooth integration
        P_S_interp = interpolate.interp1d(
            k_values, P_S_k,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Compute C_ℓ for each multipole
        C_ell = np.zeros(len(ell_values))
        
        print("Computing C_ℓ spectrum...")
        for i, ell in enumerate(ell_values):
            if ell % 100 == 0:
                print(f"  ℓ = {ell}")
            
            # Get transfer function
            Delta_ell = self.transfer.compute(ell, k_values)
            
            # Integrand: k² P_S(k) |Δ_ℓ(k)|²
            integrand = k_values**2 * P_S_k * np.abs(Delta_ell)**2
            
            # Integrate
            C_ell[i] = (2.0 / np.pi) * np.trapz(integrand, k_values)
        
        return ell_values, C_ell
    
    def flat_sky_approximation(self,
                               k_values: np.ndarray,
                               P_S_k: np.ndarray,
                               ell_values: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute C_ℓ using flat-sky approximation (high-ℓ limit).
        
        In flat sky: ℓ ≈ k χ_star, so:
            C_ℓ ≈ (2/π χ_star²) ℓ² P_S(ℓ/χ_star)
        
        This is much faster but only accurate for ℓ > ~50.
        
        Args:
            k_values: Source wavenumbers
            P_S_k: Source power spectrum
            ell_values: Multipoles to compute
            
        Returns:
            ell_values: Multipole array
            C_ell: Angular power spectrum
        """
        if ell_values is None:
            ell_values = np.arange(50, self.ell_max + 1)
        
        chi_star = self.params.chi_star
        
        # Interpolate P_S
        P_S_interp = interpolate.interp1d(
            k_values, P_S_k,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Map ℓ → k
        k_from_ell = ell_values / chi_star
        
        # Evaluate P_S at these k values
        P_S_at_ell = P_S_interp(k_from_ell)
        
        # Apply flat-sky formula
        C_ell = (2.0 / (np.pi * chi_star**2)) * ell_values**2 * P_S_at_ell
        
        return ell_values, C_ell


def load_planck_data(data_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Planck binned TT power spectrum data.
    
    Args:
        data_file: Path to data file (uses synthetic if None)
        
    Returns:
        ell: Multipole values
        C_ell: Power spectrum values (μK²)
        sigma_ell: Uncertainties (μK²)
    """
    if data_file is None:
        # Generate synthetic Planck-like data
        print("Generating synthetic Planck-like data...")
        return generate_synthetic_planck()
    
    # Load actual data
    data = np.loadtxt(data_file)
    ell = data[:, 0]
    C_ell = data[:, 1]
    sigma_ell = data[:, 2] if data.shape[1] > 2 else np.ones_like(C_ell) * 0.1 * C_ell
    
    return ell, C_ell, sigma_ell


def generate_synthetic_planck() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic Planck-like TT power spectrum.
    
    Returns:
        ell: Multipole values
        C_ell: Power spectrum (μK²)
        sigma_ell: Uncertainties (μK²)
    """
    ell = np.arange(2, 2001)
    
    # Approximate acoustic peaks with damped oscillation
    ell_peak = 220.0  # First peak location
    oscillation = np.cos(np.pi * ell / ell_peak)
    
    # Envelope (Sachs-Wolfe plateau + damping)
    envelope = 5000.0 * np.exp(-((ell - 200) / 1000)**2) * (ell / 10)**(-0.5)
    
    # Combine
    C_ell = envelope * (1.0 + 0.5 * oscillation)
    
    # Add realistic features
    C_ell[ell < 10] *= (ell[ell < 10] / 10)**2  # Low-ℓ rise
    C_ell[ell > 1000] *= np.exp(-(ell[ell > 1000] - 1000) / 500)  # High-ℓ damping
    
    # Uncertainties (cosmic variance + noise)
    sigma_ell = np.sqrt(2.0 / (2 * ell + 1)) * C_ell + 100.0
    
    return ell, C_ell, sigma_ell


def compute_chi_squared(C_ell_model: np.ndarray,
                       C_ell_data: np.ndarray,
                       sigma_ell: np.ndarray) -> float:
    """
    Compute χ² goodness-of-fit.
    
    Args:
        C_ell_model: Model prediction
        C_ell_data: Observed data
        sigma_ell: Uncertainties
        
    Returns:
        χ²/ν (reduced chi-squared)
    """
    residuals = (C_ell_model - C_ell_data) / sigma_ell
    chi_squared = np.sum(residuals**2)
    nu = len(C_ell_data)
    
    return chi_squared / nu


# ============================================================================
# Example Usage
# ============================================================================

def example_thin_shell_synthesis():
    """Example: Full spherical projection with thin shell."""
    print("\nExample 1: Thin Shell Synthesis")
    print("-" * 50)
    
    # Create synthetic source spectrum with peaks
    k = np.linspace(0.01, 2.0, 200)
    
    # Multiple Gaussian peaks (simulating KRAM resonances)
    P_S = np.zeros_like(k)
    peak_locations = [0.3, 0.7, 1.2]
    for k_peak in peak_locations:
        P_S += np.exp(-((k - k_peak) / 0.1)**2)
    
    # Synthesize
    synthesizer = CMBSynthesizer(thin_shell=True, ell_max=1000)
    ell, C_ell = synthesizer.source_to_angular_spectrum(k, P_S)
    
    print(f"Computed C_ℓ for ℓ = {ell[0]} to {ell[-1]}")
    print(f"Peak C_ℓ value: {np.max(C_ell):.3e}")
    print(f"Peak location: ℓ = {ell[np.argmax(C_ell)]}")
    
    return synthesizer, ell, C_ell


def example_flat_sky_comparison():
    """Example: Compare full vs flat-sky approximation."""
    print("\nExample 2: Flat Sky Comparison")
    print("-" * 50)
    
    # Source spectrum
    k = np.linspace(0.01, 2.0, 200)
    P_S = np.exp(-k) + 0.5 * np.cos(10 * k) * np.exp(-k)
    
    synthesizer = CMBSynthesizer(thin_shell=True, ell_max=500)
    
    # Full calculation
    ell_full, C_ell_full = synthesizer.source_to_angular_spectrum(k, P_S)
    
    # Flat sky
    ell_flat, C_ell_flat = synthesizer.flat_sky_approximation(k, P_S)
    
    # Compare at high ℓ
    common_ell = ell_full[ell_full >= 50]
    idx_full = np.isin(ell_full, common_ell)
    idx_flat = np.isin(ell_flat, common_ell)
    
    diff = np.abs(C_ell_full[idx_full] - C_ell_flat[idx_flat]) / C_ell_full[idx_full]
    print(f"Mean relative difference at ℓ > 50: {np.mean(diff):.3%}")
    
    return synthesizer, ell_full, C_ell_full, ell_flat, C_ell_flat


def example_planck_comparison():
    """Example: Compare synthetic KRAM spectrum with Planck data."""
    print("\nExample 3: Planck Comparison")
    print("-" * 50)
    
    # Load Planck data
    ell_planck, C_ell_planck, sigma_planck = load_planck_data()
    
    # Create KRAM source with multiple resonances
    k = np.linspace(0.005, 1.0, 300)
    
    # Simulate KRAM peaks (would come from actual KRAM evolution)
    P_S = np.zeros_like(k)
    chi_star = 14000.0
    
    # Place peaks at k values that map to observed Planck peaks
    planck_peaks = [220, 540, 800, 1050]
    for ell_peak in planck_peaks:
        k_peak = ell_peak / chi_star
        if k_peak < np.max(k):
            P_S += 2.0 * np.exp(-((k - k_peak) / 0.01)**2)
    
    # Synthesize
    synthesizer = CMBSynthesizer(thin_shell=True, ell_max=1200)
    ell_model, C_ell_model_raw = synthesizer.source_to_angular_spectrum(k, P_S)
    
    # Scale to match Planck amplitude (would be derived from theory)
    scale_factor = np.max(C_ell_planck) / np.max(C_ell_model_raw)
    C_ell_model = C_ell_model_raw * scale_factor
    
    # Interpolate model to Planck ℓ values
    C_ell_model_interp = np.interp(ell_planck, ell_model, C_ell_model)
    
    # Compute fit quality
    chi2_nu = compute_chi_squared(C_ell_model_interp, C_ell_planck, sigma_planck)
    
    print(f"χ²/ν = {chi2_nu:.2f}")
    print(f"Scale factor applied: {scale_factor:.2e}")
    
    return ell_model, C_ell_model, ell_planck, C_ell_planck, sigma_planck


if __name__ == "__main__":
    print("=" * 70)
    print("CMB Synthesis Module - Test Suite")
    print("=" * 70)
    
    # Run examples
    synthesizer1, ell1, C_ell1 = example_thin_shell_synthesis()
    synthesizer2, ell_f, C_f, ell_flat, C_flat = example_flat_sky_comparison()
    ell_m, C_m, ell_p, C_p, sig_p = example_planck_comparison()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
    print("\nNote: For visualization, use matplotlib to plot results:")
    print("  plt.plot(ell_model, C_ell_model, label='KUT Model')")
    print("  plt.errorbar(ell_planck, C_ell_planck, sigma_planck, label='Planck')")
