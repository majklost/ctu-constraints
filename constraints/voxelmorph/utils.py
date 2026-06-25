from collections.abc import Sequence
from typing import Literal, Union

import torch

from . import functional as vxm


def random_disp(
    shape: Sequence[int],
    scales: Union[float, int, list[float]] = 10,
    magnitude: float = 10,
    integrations: int = 0,
    voxsize: float = 1,
    meshgrid: Union[torch.Tensor, None] = None,
    device: Union[torch.device, None] = None,
    fractal_mode: Literal["blur", "upsample"] = "upsample",
) -> torch.Tensor:
    """
    Generate random displacement field for images in (B, C, *spatial) format.

    Takes shape in (B, C, *spatial) format (matching image tensors) and outputs
    displacement field in (B, ndim, *spatial) format - channels-first format.
    The channel dimension is ignored since displacement is per-voxel, not per-channel.

    Parameters
    ----------
    shape : Sequence[int]
        Shape in (B, C, *spatial) format matching the image to be transformed.
        Examples: (1, 1, 64, 64) for 2D, (2, 3, 64, 64, 64) for 3D.
    scales : float, int, or List[float], default=10
        Smoothing scale(s) for fractal noise, divided by voxsize. Interpretation depends
        on fractal_mode:
        - fractal_mode='blur': sigma values for Gaussian smoothing
        - fractal_mode='upsample': downsampling factors for upsampled noise
    magnitude : float, default=10
        Standard deviation of displacement in voxel coordinates, divided by voxsize.
    integrations : int, default=0
        Number of integration steps for diffeomorphic transform. If 0, no integration.
    voxsize : float, default=1
        Voxel size for scaling smoothing and magnitude parameters.
    meshgrid : torch.Tensor or None, default=None
        Coordinate grid of shape (ndim, *spatial) for integration. If None and
        integrations > 0, computed internally.
    device : torch.device or None, default=None
        Device for tensor allocation.
    fractal_mode : {'blur', 'upsample'}, default='upsample'
        Fractal noise generation method:
        - 'blur': Generate noise and apply Gaussian smoothing (higher quality)
        - 'upsample': Generate coarse noise and upsample (faster, lower memory)

    Returns
    -------
    torch.Tensor
        Displacement field with shape (B, ndim, *spatial) - channels-first format.

    Examples
    --------
    >>> # Generate displacement for 2D image with shape (B, C, H, W)
    >>> disp = random_disp(shape=(1, 1, 64, 64), scales=5.0, magnitude=3.0)
    >>> disp.shape
    torch.Size([1, 2, 64, 64])

    >>> # Generate displacement for 3D image with shape (B, C, D, H, W)
    >>> disp = random_disp(shape=(2, 3, 32, 32, 32), integrations=5)
    >>> disp.shape
    torch.Size([2, 3, 32, 32, 32])
    """
    # Extract batch and spatial shape, ignoring channel dimension
    batch_size = shape[0]
    spatial_shape = shape[2:]  # Skip B and C

    return vxm.random_disp(
        shape=(batch_size, *spatial_shape),
        scales=scales,
        magnitude=magnitude,
        integrations=integrations,
        voxsize=voxsize,
        meshgrid=meshgrid,
        non_spatial_dims=(0,),
        device=device,
        fractal_mode=fractal_mode,
    )


def spatial_transform(
    image: torch.Tensor,
    trf: Union[torch.Tensor, None],
    method: Literal["nearest", "linear"] = "linear",
    isdisp: bool = True,
    meshgrid: Union[torch.Tensor, None] = None,
    origin_at_center: bool = True,
) -> torch.Tensor:
    """
    Apply spatial transformation to image in (B, C, *spatial) format.

    Wrapper around voxelmorph.functional.spatial_transform with non_spatial_dims=(0, 1).

    Parameters
    ----------
    image : torch.Tensor
        Input image with shape (B, C, *spatial).
    trf : torch.Tensor or None
        Transformation field. Can be:
        - Affine matrix: shape (N+1, N+1) or (N, N+1)
        - Displacement field: shape (N, *spatial) - channels-first format
        - None: returns image unchanged
    method : str, default='linear'
        Interpolation mode ('linear' or 'nearest').
    isdisp : bool, default=True
        If True, treat trf as displacement field (N, *spatial). If False, treat as
        coordinates (*spatial, N) ready for grid_sample.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial).
    origin_at_center : bool, default=True
        Place origin at image center for affine transformations.

    Returns
    -------
    torch.Tensor
        Transformed image with shape (B, C, *spatial).

    Examples
    --------
    >>> # 2D image with batch and channel
    >>> image = torch.randn(2, 3, 64, 64)
    >>> disp = torch.randn(2, 64, 64)  # (ndim, H, W)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([2, 3, 64, 64])

    >>> # 3D image with batch and channel
    >>> image = torch.randn(1, 1, 64, 64, 64)
    >>> disp = torch.randn(3, 64, 64, 64)  # (ndim, D, H, W)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([1, 1, 64, 64, 64])
    """
    return vxm.spatial_transform(
        image=image,
        trf=trf,
        mode=method,
        isdisp=isdisp,
        meshgrid=meshgrid,
        origin_at_center=origin_at_center,
        non_spatial_dims=(0, 1),
        align_corners=True,
    )
