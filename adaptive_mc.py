"""
Adaptive Marching Cubes for TripoSR
=====================================
Coarse-to-fine approach: sample at low resolution to find surface regions,
then only run high-resolution marching cubes where the surface actually is.

Typical speedup: 2-4x over full-grid MC for furniture-scale objects.

Adapted from friend's HybridAdaptiveMarchingCubes to work with
TripoSR's query_triplane API and coordinate conventions.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import trimesh
from torchmcubes import marching_cubes


class AdaptiveMarchingCubes:
    """
    Coarse-to-fine marching cubes.

    Steps:
      1. Query density at coarse_resolution³ grid (cheap)
      2. Find cells where density crosses the surface threshold (sign change)
      3. Compute tight bounding box around active cells + padding
      4. Build a fine grid ONLY inside that bounding box
      5. Query density at fine_resolution only for those points
      6. Run marching cubes on the fine sub-grid
      7. Map vertices back to world coordinates
    """

    points_range: Tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        resolution: int = 256,
        coarse_resolution: int = 64,
        padding: int = 2,
    ) -> None:
        self.resolution = resolution
        self.coarse_resolution = coarse_resolution
        self.padding = padding

        self._fine_cell = (self.points_range[1] - self.points_range[0]) / self.resolution
        self._coarse_grid: Optional[torch.Tensor] = None

    @property
    def _coarse_vertices(self) -> torch.Tensor:
        """Pre-computed coarse grid in (0,1) space — built once and reused."""
        if self._coarse_grid is None:
            D = self.coarse_resolution
            x = torch.linspace(*self.points_range, D)
            y = torch.linspace(*self.points_range, D)
            z = torch.linspace(*self.points_range, D)
            gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
            self._coarse_grid = torch.stack([gx, gy, gz], dim=-1)
        return self._coarse_grid

    @torch.no_grad()
    def extract(
        self,
        query_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        threshold: float = 25.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_fn: Takes positions (N, 3) in (0,1) space, returns density_act (N,).
            device: CUDA device.
            threshold: Density iso-surface threshold (TripoSR default is 25.0).

        Returns:
            v_pos: (V, 3) vertex positions in (0,1) space.
            t_idx: (F, 3) face indices.
        """
        D = self.coarse_resolution

        # ── 1. Coarse density query ────────────────────────────────────────
        coarse_pts = self._coarse_vertices.reshape(-1, 3).to(device)
        coarse_density = query_fn(coarse_pts).view(D, D, D)

        # ── 2. Find active cells (sign change relative to threshold) ───────
        c = coarse_density
        min_d = torch.minimum(
            torch.minimum(torch.minimum(c[:-1,:-1,:-1], c[:-1,:-1,1:]),
                          torch.minimum(c[:-1,1:,:-1],  c[:-1,1:,1:])),
            torch.minimum(torch.minimum(c[1:,:-1,:-1],  c[1:,:-1,1:]),
                          torch.minimum(c[1:,1:,:-1],   c[1:,1:,1:])),
        )
        max_d = torch.maximum(
            torch.maximum(torch.maximum(c[:-1,:-1,:-1], c[:-1,:-1,1:]),
                          torch.maximum(c[:-1,1:,:-1],  c[:-1,1:,1:])),
            torch.maximum(torch.maximum(c[1:,:-1,:-1],  c[1:,:-1,1:]),
                          torch.maximum(c[1:,1:,:-1],   c[1:,1:,1:])),
        )
        active = (min_d < threshold) & (max_d > threshold)

        if not active.any():
            return (
                torch.empty(0, 3, device=device),
                torch.empty(0, 3, dtype=torch.long, device=device),
            )

        # ── 3. Tight bounding box ──────────────────────────────────────────
        idx = torch.nonzero(active)
        lo = (idx.min(0).values - self.padding).clamp(min=0)
        hi = (idx.max(0).values + 1 + self.padding).clamp(max=D - 1)

        # ── 4. Map coarse bbox → fine grid ────────────────────────────────
        scale = self.resolution // self.coarse_resolution
        fine_lo = lo * scale
        fine_hi = hi * scale + 1
        fine_dims = (fine_hi - fine_lo).tolist()

        cell = self._fine_cell
        p0 = self.points_range[0]

        axes = [
            torch.linspace(
                p0 + fine_lo[i].item() * cell,
                p0 + (fine_hi[i].item() - 1) * cell,
                fine_dims[i],
                device=device,
            )
            for i in range(3)
        ]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        fine_pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

        coverage = fine_pts.shape[0] / (self.resolution ** 3)
        print(f"  Adaptive MC: querying {fine_pts.shape[0]:,} / {self.resolution**3:,} pts ({coverage:.1%} of full grid)")

        # ── 5. Fine density query ──────────────────────────────────────────
        fine_density = query_fn(fine_pts)
        # TripoSR convention: surface where density_act = threshold
        # MC iso = 0, so pass -(density - threshold)
        level = -(fine_density - threshold).view(*fine_dims)

        # ── 6. Marching cubes ──────────────────────────────────────────────
        try:
            v_pos, t_idx = marching_cubes(level, 0.0)
        except Exception:
            v_pos, t_idx = marching_cubes(level.cpu(), 0.0)
            v_pos = v_pos.to(device)
            t_idx = t_idx.to(device)

        if v_pos.shape[0] == 0:
            return v_pos, t_idx

        # ── 7. Map vertices back to (0,1) world space ─────────────────────
        # torchmcubes returns indices in [0, dim-1], axis order z,y,x
        v_pos = v_pos[..., [2, 1, 0]]
        dims_t = torch.tensor(fine_dims, device=v_pos.device, dtype=v_pos.dtype)
        # normalise to [0,1] within the sub-grid
        v_pos = v_pos / (dims_t - 1)
        # scale and offset to (0,1) global space
        scale_vec = torch.tensor(
            [(fine_dims[i] - 1) * cell for i in range(3)],
            device=v_pos.device, dtype=v_pos.dtype,
        )
        offset = torch.tensor(
            [p0 + fine_lo[i].item() * cell for i in range(3)],
            device=v_pos.device, dtype=v_pos.dtype,
        )
        v_pos = v_pos * scale_vec + offset

        return v_pos, t_idx


def extract_mesh_adaptive(
    model,
    scene_code: torch.Tensor,
    device,
    resolution: int = 256,
    coarse_resolution: int = 64,
    threshold: float = 25.0,
    has_vertex_color: bool = False,
) -> trimesh.Trimesh:
    """
    Drop-in replacement for model.extract_mesh() that uses adaptive MC.

    Returns a trimesh.Trimesh in the same coordinate space as the standard
    extract_mesh (world coords scaled to the renderer radius).
    """
    from tsr.utils import scale_tensor

    radius = model.renderer.cfg.radius
    amc = AdaptiveMarchingCubes(resolution=resolution, coarse_resolution=coarse_resolution)

    def query_fn(positions_01: torch.Tensor) -> torch.Tensor:
        """Converts (0,1) → (-radius, radius), queries triplane, returns density_act."""
        positions_world = scale_tensor(
            positions_01,
            (0.0, 1.0),
            (-radius, radius),
        )
        with torch.no_grad():
            out = model.renderer.query_triplane(
                model.decoder,
                positions_world,
                scene_code,
            )
        return out["density_act"].squeeze(-1)

    v_pos_01, t_idx = amc.extract(query_fn, device=device, threshold=threshold)

    if v_pos_01.shape[0] == 0:
        return trimesh.Trimesh()

    # Scale vertices from (0,1) → (-radius, radius) world space
    v_pos_world = scale_tensor(v_pos_01, (0.0, 1.0), (-radius, radius))

    color = None
    if has_vertex_color:
        with torch.no_grad():
            color = model.renderer.query_triplane(
                model.decoder, v_pos_world, scene_code,
            )["color"]

    mesh = trimesh.Trimesh(
        vertices=v_pos_world.cpu().numpy(),
        faces=t_idx.cpu().numpy(),
        vertex_colors=(color.cpu().numpy() if color is not None else None),
        process=False,
    )
    return mesh
