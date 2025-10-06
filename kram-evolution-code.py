"""
KRAM Evolution Solver
=====================

Implements the relaxational PDE solver for the KnoWellian Resonant Attractor 
Manifold (KRAM) field dynamics.

The evolution equation:
    τ_M ∂g_M/∂t = ξ² ∇²g_M - μ² g_M - β g_M³ + J_imprint + η

This is a driven, damped, nonlinear field equation (Allen-Cahn/Ginzburg-Landau 
type) where the manifold "learns" from incoming imprints while deepening stable 
attractor patterns.

Author: David Noel Lynch
Date: 2025
License: MIT
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class KRAMParameters:
    """
    Physical and numerical parameters for KRAM evolution.
    
    Attributes:
        tau_M: Manifold relaxation timescale
        xi_squared: Stiffness parameter (penalizes high curvature)
        mu_squared: Mass-like parameter (controls baseline stability)
        beta: Nonlinear saturation coefficient (creates attractor wells)
        kappa: Coupling strength to imprint current
        noise_amplitude: Amplitude of stochastic fluctuations
        dt: Time step for integration
        dx: Spatial grid spacing
        clip_value: Maximum absolute value for field (numerical stability)
    """
    tau_M: float = 1.0
    xi_squared: float = 0.1
    mu_squared: float = 0.1
    beta: float = 1.0
    kappa: float = 1.0
    noise_amplitude: float = 0.01
    dt: float = 0.01
    dx: float = 1.0
    clip_value: float = 10.0
    
    def effective_mass_squared(self, k: np.ndarray) -> np.ndarray:
        """
        Compute effective mass for wavenumber k.
        
        m_eff²(k) = k² ξ² + μ²
        """
        return k**2 * self.xi_squared + self.mu_squared


class KRAMSolver:
    """
    Solver for KRAM field evolution using IMEX (Implicit-Explicit) timestepping.
    
    The PDE is split into:
    - Stiff part (Laplacian): treated implicitly
    - Non-stiff parts (nonlinear, forcing): treated explicitly
    """
    
    def __init__(self, 
                 grid_shape: Tuple[int, ...],
                 params: Optional[KRAMParameters] = None):
        """
        Initialize KRAM solver.
        
        Args:
            grid_shape: Shape of spatial grid (e.g., (64, 64) for 2D)
            params: Physical parameters (uses defaults if None)
        """
        self.grid_shape = grid_shape
        self.ndim = len(grid_shape)
        self.params = params or KRAMParameters()
        
        # Initialize field
        self.g_M = np.zeros(grid_shape)
        
        # Precompute Laplacian operator in Fourier space
        self._setup_fourier_laplacian()
        
        # Time tracking
        self.t = 0.0
        self.step_count = 0
        
    def _setup_fourier_laplacian(self):
        """
        Precompute the Fourier-space Laplacian operator for efficient solving.
        
        For periodic boundary conditions:
        ∇² → -k² in Fourier space
        """
        k_grids = []
        for i, N in enumerate(self.grid_shape):
            k = 2 * np.pi * np.fft.fftfreq(N, d=self.params.dx)
            k_grids.append(k)
        
        # Build full k-space grid
        k_arrays = np.meshgrid(*k_grids, indexing='ij')
        self.k_squared = sum(k**2 for k in k_arrays)
        
        # Implicit operator: (1 + dt/τ_M * (ξ² k² + μ²))
        self.implicit_factor = 1.0 / (1.0 + (self.params.dt / self.params.tau_M) * 
                                       (self.params.xi_squared * self.k_squared + 
                                        self.params.mu_squared))
    
    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute spatial Laplacian using FFT for periodic boundaries.
        
        Args:
            field: Input field
            
        Returns:
            ∇²field
        """
        field_fft = np.fft.fftn(field)
        laplacian_fft = -self.k_squared * field_fft
        return np.real(np.fft.ifftn(laplacian_fft))
    
    def nonlinear_term(self, field: np.ndarray) -> np.ndarray:
        """
        Compute nonlinear saturation term: -β g³
        
        Args:
            field: Input field g_M
            
        Returns:
            -β g_M³
        """
        return -self.params.beta * field**3
    
    def add_noise(self) -> np.ndarray:
        """
        Generate spatially-correlated stochastic noise.
        
        Returns:
            Gaussian noise field with amplitude scaled by noise_amplitude
        """
        noise = np.random.randn(*self.grid_shape)
        # Apply light smoothing for spatial correlation
        noise = ndimage.gaussian_filter(noise, sigma=1.0)
        return self.params.noise_amplitude * noise
    
    def step(self, 
             J_imprint: Optional[np.ndarray] = None,
             explicit_update: Optional[Callable] = None) -> np.ndarray:
        """
        Advance KRAM field by one timestep using IMEX scheme.
        
        Split: τ_M ∂g/∂t = [ξ² ∇² - μ²]g + [-β g³ + J + η]
                            ^implicit^      ^explicit^
        
        Args:
            J_imprint: Imprint current (forcing term)
            explicit_update: Optional custom function for explicit terms
            
        Returns:
            Updated g_M field
        """
        if J_imprint is None:
            J_imprint = np.zeros_like(self.g_M)
        
        # Explicit terms
        nonlinear = self.nonlinear_term(self.g_M)
        noise = self.add_noise()
        
        if explicit_update is not None:
            explicit_terms = explicit_update(self.g_M, self.t)
        else:
            explicit_terms = nonlinear + J_imprint + noise
        
        # Forward Euler for explicit part
        g_star = self.g_M + (self.params.dt / self.params.tau_M) * explicit_terms
        
        # Implicit solve for diffusion terms via FFT
        # (1 + dt/τ * (ξ²∇² + μ²))g^{n+1} = g*
        # In Fourier space: (1 + dt/τ * (ξ²k² + μ²))ĝ^{n+1} = ĝ*
        g_star_fft = np.fft.fftn(g_star)
        g_new_fft = g_star_fft * self.implicit_factor
        g_new = np.real(np.fft.ifftn(g_new_fft))
        
        # Clip for numerical stability
        g_new = np.clip(g_new, -self.params.clip_value, self.params.clip_value)
        
        # Update state
        self.g_M = g_new
        self.t += self.params.dt
        self.step_count += 1
        
        return self.g_M
    
    def evolve(self,
               n_steps: int,
               J_imprint_func: Optional[Callable[[float], np.ndarray]] = None,
               callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
               ) -> np.ndarray:
        """
        Evolve KRAM field for multiple timesteps.
        
        Args:
            n_steps: Number of time steps
            J_imprint_func: Function J(t) returning imprint current at time t
            callback: Optional function called each step as callback(step, time, field)
            
        Returns:
            Final g_M field
        """
        for i in range(n_steps):
            # Compute imprint current for this timestep
            if J_imprint_func is not None:
                J = J_imprint_func(self.t)
            else:
                J = None
            
            # Step forward
            self.step(J_imprint=J)
            
            # User callback
            if callback is not None:
                callback(self.step_count, self.t, self.g_M)
        
        return self.g_M
    
    def compute_power_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute isotropic power spectrum of current g_M field.
        
        Returns:
            k_bins: Radial wavenumbers
            P_k: Power spectrum P(k)
        """
        # FFT of field
        g_fft = np.fft.fftn(self.g_M)
        power = np.abs(g_fft)**2
        
        # Compute radial k
        k_grids = []
        for i, N in enumerate(self.grid_shape):
            k = 2 * np.pi * np.fft.fftfreq(N, d=self.params.dx)
            k_grids.append(k)
        k_arrays = np.meshgrid(*k_grids, indexing='ij')
        k_radial = np.sqrt(sum(k**2 for k in k_arrays))
        
        # Bin by k
        k_flat = k_radial.flatten()
        power_flat = power.flatten()
        
        # Create bins
        k_max = np.max(k_flat)
        n_bins = min(50, self.grid_shape[0] // 2)
        k_bins = np.linspace(0, k_max, n_bins + 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        
        # Average power in each bin
        P_k = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
            if np.any(mask):
                P_k[i] = np.mean(power_flat[mask])
        
        return k_centers, P_k
    
    def reset(self, initial_field: Optional[np.ndarray] = None):
        """
        Reset solver to initial state.
        
        Args:
            initial_field: Optional initial condition (zeros if None)
        """
        if initial_field is not None:
            self.g_M = initial_field.copy()
        else:
            self.g_M = np.zeros(self.grid_shape)
        self.t = 0.0
        self.step_count = 0


def create_gaussian_imprint(center: Tuple[float, ...],
                            amplitude: float,
                            width: float,
                            grid_shape: Tuple[int, ...],
                            dx: float = 1.0) -> np.ndarray:
    """
    Create a Gaussian imprint (localized forcing).
    
    Args:
        center: Center coordinates (in grid units)
        amplitude: Peak amplitude
        width: Gaussian width (sigma)
        grid_shape: Shape of output grid
        dx: Grid spacing
        
    Returns:
        Gaussian imprint field
    """
    # Create coordinate grids
    coords = [np.arange(N) * dx for N in grid_shape]
    grids = np.meshgrid(*coords, indexing='ij')
    
    # Compute distance from center
    r_squared = sum((g - c*dx)**2 for g, c in zip(grids, center))
    
    # Gaussian
    return amplitude * np.exp(-r_squared / (2 * width**2))


def create_hex_lattice_imprint(amplitude: float,
                               wavelength: float,
                               grid_shape: Tuple[int, int],
                               dx: float = 1.0,
                               phase: float = 0.0) -> np.ndarray:
    """
    Create hexagonal lattice pattern (simplified Cairo Q-Lattice approximation).
    
    Args:
        amplitude: Pattern amplitude
        wavelength: Fundamental wavelength
        grid_shape: Shape of 2D grid
        dx: Grid spacing
        phase: Global phase shift
        
    Returns:
        Hexagonal pattern field
    """
    assert len(grid_shape) == 2, "Hex lattice only for 2D grids"
    
    x = np.arange(grid_shape[0]) * dx
    y = np.arange(grid_shape[1]) * dx
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    k = 2 * np.pi / wavelength
    
    # Three plane waves at 120° create hexagonal pattern
    pattern = (np.cos(k * X + phase) + 
               np.cos(k * (0.5 * X - np.sqrt(3)/2 * Y) + phase) +
               np.cos(k * (0.5 * X + np.sqrt(3)/2 * Y) + phase))
    
    return amplitude * pattern / 3.0


# ============================================================================
# Example Usage and Tests
# ============================================================================

def example_relaxation():
    """
    Example: Field relaxation from random initial condition.
    """
    print("Example 1: Field Relaxation")
    print("-" * 50)
    
    # Setup
    params = KRAMParameters(
        tau_M=1.0,
        xi_squared=0.5,
        mu_squared=0.1,
        beta=1.0,
        dt=0.01
    )
    
    solver = KRAMSolver(grid_shape=(64, 64), params=params)
    
    # Random initial condition
    solver.g_M = np.random.randn(64, 64) * 0.5
    
    print(f"Initial energy: {np.mean(solver.g_M**2):.4f}")
    
    # Evolve
    solver.evolve(n_steps=1000)
    
    print(f"Final energy: {np.mean(solver.g_M**2):.4f}")
    print(f"Final time: {solver.t:.2f}")
    
    return solver


def example_driven_evolution():
    """
    Example: Evolution with localized particle imprint.
    """
    print("\nExample 2: Driven Evolution with Particle")
    print("-" * 50)
    
    params = KRAMParameters(
        tau_M=1.0,
        xi_squared=0.2,
        mu_squared=-0.1,  # Negative for pattern formation
        beta=1.0,
        kappa=0.5,
        dt=0.01
    )
    
    solver = KRAMSolver(grid_shape=(64, 64), params=params)
    
    # Create static particle imprint
    particle = create_gaussian_imprint(
        center=(32, 32),
        amplitude=2.0,
        width=3.0,
        grid_shape=(64, 64)
    )
    
    # Create hex lattice (vacuum structure)
    vacuum = create_hex_lattice_imprint(
        amplitude=0.3,
        wavelength=10.0,
        grid_shape=(64, 64)
    )
    
    # Combined forcing
    J_total = particle + vacuum
    
    def forcing(t):
        return params.kappa * J_total
    
    # Evolve
    solver.evolve(n_steps=2000, J_imprint_func=forcing)
    
    print(f"Final time: {solver.t:.2f}")
    print(f"Pattern amplitude: {np.max(np.abs(solver.g_M)):.4f}")
    
    # Compute power spectrum
    k, P_k = solver.compute_power_spectrum()
    peaks = k[P_k > 0.5 * np.max(P_k)]
    print(f"Spectral peaks near k = {peaks[:5]}")
    
    return solver, k, P_k


def example_time_dependent_forcing():
    """
    Example: Coherent pumping (Control) + incoherent chaos.
    """
    print("\nExample 3: Control-Chaos Balance")
    print("-" * 50)
    
    params = KRAMParameters(
        tau_M=1.0,
        xi_squared=0.3,
        mu_squared=0.0,
        beta=2.0,
        dt=0.01,
        noise_amplitude=0.05
    )
    
    solver = KRAMSolver(grid_shape=(64, 64), params=params)
    
    # Vacuum structure
    vacuum = create_hex_lattice_imprint(
        amplitude=0.2,
        wavelength=8.0,
        grid_shape=(64, 64)
    )
    
    # Time-dependent forcing
    omega_pump = 0.5  # Pump frequency
    pump_amplitude = 1.5
    chaos_strength = 1.0
    
    def forcing(t):
        # Coherent pump (Control)
        pump = pump_amplitude * np.cos(omega_pump * t) * vacuum
        
        # Incoherent chaos (noise added by solver automatically)
        # Additional structured chaos
        chaos = chaos_strength * np.random.randn(64, 64) * 0.1
        chaos = ndimage.gaussian_filter(chaos, sigma=2.0)
        
        return pump + chaos
    
    # Evolve
    history = []
    
    def callback(step, t, field):
        if step % 100 == 0:
            history.append(field.copy())
    
    solver.evolve(n_steps=5000, J_imprint_func=forcing, callback=callback)
    
    print(f"Final time: {solver.t:.2f}")
    print(f"Number of snapshots: {len(history)}")
    
    return solver, history


def visualize_results(solver: KRAMSolver, 
                     k: Optional[np.ndarray] = None,
                     P_k: Optional[np.ndarray] = None):
    """
    Create visualization of KRAM field and power spectrum.
    """
    fig, axes = plt.subplots(1, 2 if P_k is None else 3, 
                             figsize=(15 if P_k is None else 18, 5))
    
    # Field visualization
    im = axes[0].imshow(solver.g_M, cmap='RdBu_r', 
                       vmin=-np.max(np.abs(solver.g_M)),
                       vmax=np.max(np.abs(solver.g_M)))
    axes[0].set_title(f'g_M field at t={solver.t:.2f}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im, ax=axes[0])
    
    # Power spectrum (if not provided, compute it)
    if k is None:
        k, P_k = solver.compute_power_spectrum()
    
    axes[1].semilogy(k, P_k, 'b-', linewidth=2)
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Power P(k)')
    axes[1].set_title('Power Spectrum')
    axes[1].grid(True, alpha=0.3)
    
    if len(axes) > 2:
        # Additional diagnostics
        axes[2].hist(solver.g_M.flatten(), bins=50, alpha=0.7)
        axes[2].set_xlabel('g_M value')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Field Distribution')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("KRAM Evolution Solver - Test Suite")
    print("=" * 70)
    
    # Run examples
    solver1 = example_relaxation()
    solver2, k2, P_k2 = example_driven_evolution()
    solver3, history3 = example_time_dependent_forcing()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
    
    # Optionally visualize (uncomment to display)
    # fig1 = visualize_results(solver1)
    # fig2 = visualize_results(solver2, k2, P_k2)
    # fig3 = visualize_results(solver3)
    # plt.show()
