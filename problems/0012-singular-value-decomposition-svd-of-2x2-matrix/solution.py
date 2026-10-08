import torch

def svd_2x2_singular_values(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 torch tensor
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """
    # Your code here
    A = torch.as_tensor(A, dtype = torch.float32)
    U, S, Vt = torch.linalg.svd(A)
    return U, S, Vt