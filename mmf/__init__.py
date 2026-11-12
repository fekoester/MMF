"""
Master memory function (MMF) for delay-based reservoir computers with single-variable dynamics.

    from mmf import master_memory_function, simulate_memory_capacity, scan
    C = master_memory_function(a=-0.5, b=0.4, theta=0.5, N=100, tau_in_theta=141)   # C_l per recall step
    C.sum()                                                                          # total linear MC

simulate_memory_capacity runs the linear map itself and measures C_l by regression (the ground truth the
MMF approximates); scan evaluates the MMF over a grid of one parameter.
"""

from .core import master_memory_function, modified_state_matrix, scan, simulate_memory_capacity

__all__ = ["master_memory_function", "modified_state_matrix", "simulate_memory_capacity", "scan"]
__version__ = "0.2.0"
