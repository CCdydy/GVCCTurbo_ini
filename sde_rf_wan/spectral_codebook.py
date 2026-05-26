"""
spectral_codebook.py — V2-a spectral codebook variants.

Stage 0 of V2-a keeps the full-resolution Wan trajectory unchanged, but makes
codebook selection frequency-aware.

Stage 0a, kept as a negative control, injects only the active DCT band and was
falsified by experiment. Stage 0b is spectrum-preserving: active-band DCT
coefficients are replaced by the selected atom, while inactive coefficients are
filled by deterministic Gaussian noise from a shared seed.

Because the DCT is orthonormal, an i.i.d. Gaussian spatial atom transformed to
DCT space is still i.i.d. Gaussian. We therefore generate atoms directly in the
active DCT coefficient subspace, which avoids expensive DCTs over every random
atom and gives the intended lower-dimensional search.
"""

import math
import os
import sys
from typing import Dict, List, Tuple

import torch

_spd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spd_repro")
if _spd_path not in sys.path:
    sys.path.insert(0, _spd_path)
from dct_utils import dct_2d, idct_2d, radial_frequency_bins


class ThreeBandStepSchedule:
    """Manual low / low+mid / full-band schedule over SDE step indices.

    Step 0 is the high-noise end. The default matches the V2-a note for
    17 codebook steps:

      steps 0-5:    low only
      steps 6-11:   low + mid
      steps 12-16:  low + mid + high
    """

    def __init__(
        self,
        low_until: int = 6,
        mid_until: int = 12,
        low_cutoff: float = 1.0 / 3.0,
        mid_cutoff: float = 2.0 / 3.0,
        soft: bool = False,
        soft_width: float = 0.02,
    ):
        assert 0 <= low_until <= mid_until
        assert 0.0 < low_cutoff <= mid_cutoff <= 1.0
        self.low_until = low_until
        self.mid_until = mid_until
        self.low_cutoff = low_cutoff
        self.mid_cutoff = mid_cutoff
        self.soft = soft
        self.soft_width = soft_width
        self.name = (
            f"three_band[0-{low_until - 1}:low,"
            f"{low_until}-{mid_until - 1}:low+mid,"
            f"{mid_until}+:all]"
        )
        self._mask_cache: Dict[Tuple, torch.Tensor] = {}

    def stage(self, sde_idx: int) -> int:
        if sde_idx < self.low_until:
            return 0
        if sde_idx < self.mid_until:
            return 1
        return 2

    def cutoff(self, sde_idx: int, t: float = None) -> float:
        stage = self.stage(sde_idx)
        if stage == 0:
            return self.low_cutoff
        if stage == 1:
            return self.mid_cutoff
        return 1.0

    def mask(self, sde_idx: int, t: float, H: int, W: int, device, dtype) -> torch.Tensor:
        cutoff = self.cutoff(sde_idx, t)
        key = (sde_idx, cutoff, H, W, str(device), str(dtype), self.soft, self.soft_width)
        if key in self._mask_cache:
            return self._mask_cache[key]

        omega_mag, _, _ = radial_frequency_bins(H, W, num_bins=max(H, W), device=device)
        if not self.soft or cutoff >= 1.0:
            m = (omega_mag <= cutoff).to(dtype)
        else:
            lo = cutoff - self.soft_width / 2
            hi = cutoff + self.soft_width / 2
            m = torch.zeros_like(omega_mag)
            m[omega_mag <= lo] = 1.0
            in_taper = (omega_mag > lo) & (omega_mag < hi)
            taper_vals = 0.5 * (
                1.0 + torch.cos(math.pi * (omega_mag[in_taper] - lo) / self.soft_width)
            )
            m[in_taper] = taper_vals
            m = m.to(dtype)
        self._mask_cache[key] = m
        return m

    def __repr__(self):
        return self.name


class SpectralBandCodebook:
    """Per-frame masked-random codebook in DCT coefficient space."""

    def __init__(
        self,
        K: int = 16384,
        M: int = 64,
        frame_shape: Tuple[int, ...] = (16, 90, 160),
        seed: int = 42,
        device: torch.device = torch.device("cuda"),
        gen_batch: int = 0,
    ):
        self.K = K
        self.M = M
        assert M <= K, f"M ({M}) must be <= K ({K})"
        self.frame_shape = frame_shape
        self.C, self.H, self.W = frame_shape
        self.seed = seed
        self.device = device
        self.bits_per_index = math.ceil(math.log2(K)) if K > 1 else 1
        self.bits_per_frame_step = M * (self.bits_per_index + 1)

        self.gen_batch = gen_batch

    def _sf_seed(self, step: int, frame: int) -> int:
        return self.seed * 100003 + step * 10007 + frame

    def _mask_weights(
        self,
        schedule: ThreeBandStepSchedule,
        step_idx: int,
        t: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return active DCT coordinates and optional taper weights."""
        weights = schedule.mask(
            step_idx, t, self.H, self.W, self.device, torch.float32,
        ).reshape(-1)
        active = weights > 0
        assert active.any(), "V2-a spectral codebook mask must keep at least one coefficient"
        return active, weights[active], weights

    def _carrier_seed(self, step: int, frame: int) -> int:
        return self.seed * 1000003 + step * 100069 + frame * 1009 + 7919

    def _batch_size(self, D_band: int) -> int:
        if self.gen_batch > 0:
            return self.gen_batch
        mem_budget = 64 * 1024 * 1024
        atom_bytes = max(1, D_band) * 4
        return max(32, min(512, mem_budget // atom_bytes))

    def active_coefficients(
        self,
        schedule: ThreeBandStepSchedule,
        step_idx: int,
        t: float = 0.5,
    ) -> Tuple[int, int, int]:
        """Return active spatial DCT coeffs, total coeffs, and lifted atom dim."""
        active, _, _ = self._mask_weights(schedule, step_idx, t)
        n_active = int(active.sum().item())
        n_total = int(active.numel())
        return n_active, n_total, n_active * self.C

    def _generate_atoms_batch(self, gen: torch.Generator, count: int, D_band: int) -> torch.Tensor:
        atoms = torch.empty(count, D_band, device=self.device)
        for i in range(count):
            atoms[i] = torch.randn(D_band, generator=gen, device=self.device)
        return atoms

    def _project_residual(
        self,
        residual: torch.Tensor,
        active_flat: torch.Tensor,
        active_weights: torch.Tensor,
    ) -> torch.Tensor:
        r_dct = dct_2d(residual.float().to(self.device))
        r_active = r_dct.reshape(self.C, -1)[:, active_flat]
        return (r_active * active_weights.unsqueeze(0)).reshape(-1)

    def _lift_to_spatial(
        self,
        vec: torch.Tensor,
        active_flat: torch.Tensor,
        active_weights: torch.Tensor,
        weights_flat: torch.Tensor,
        preserve_spectrum: bool,
        carrier_seed: int,
    ) -> torch.Tensor:
        selected = vec.reshape(self.C, -1)
        if preserve_spectrum:
            selected_std = selected.std()
            if selected_std > 1e-8:
                selected = selected / selected_std
            gen = torch.Generator(device=self.device).manual_seed(carrier_seed)
            dct_flat = torch.randn(
                self.C, self.H * self.W, generator=gen, device=self.device,
            )
            dct_flat[:, active_flat] = (
                dct_flat[:, active_flat] * (1.0 - active_weights.unsqueeze(0))
                + selected * active_weights.unsqueeze(0)
            )
        else:
            dct_flat = torch.zeros(self.C, self.H * self.W, device=self.device)
            dct_flat[:, active_flat] = selected * active_weights.unsqueeze(0)
        z = idct_2d(dct_flat.reshape(self.C, self.H, self.W))
        std = z.std()
        if std > 1e-8:
            z = z / std
        return z

    def select_atoms(
        self,
        residual: torch.Tensor,
        step_idx: int,
        frame_idx: int,
        schedule: ThreeBandStepSchedule,
        t: float,
        M_override: int = None,
        preserve_spectrum: bool = False,
    ) -> Tuple[List[int], List[int], torch.Tensor]:
        """Select top-M atoms using only the active DCT band."""
        M = M_override if M_override is not None else self.M
        active_flat, active_weights, weights_flat = self._mask_weights(schedule, step_idx, t)
        r_vec = self._project_residual(residual, active_flat, active_weights)
        D_band = r_vec.numel()
        seed_sf = self._sf_seed(step_idx, frame_idx)
        gen = torch.Generator(device=self.device).manual_seed(seed_sf)
        all_ips = torch.empty(self.K, device=self.device)
        batch = self._batch_size(D_band)

        for b0 in range(0, self.K, batch):
            bc = min(batch, self.K - b0)
            atoms = self._generate_atoms_batch(gen, bc, D_band)
            all_ips[b0:b0 + bc] = atoms @ r_vec
            del atoms

        _, top_idx = all_ips.abs().topk(M)
        top_idx_sorted = top_idx.sort().values
        signs_t = torch.sign(all_ips[top_idx_sorted])
        signs_t[signs_t == 0] = 1.0
        idx_list = top_idx_sorted.cpu().tolist()
        sign_list = signs_t.cpu().int().tolist()

        combined = self._regenerate_and_combine(
            seed_sf, idx_list, signs_t, D_band, active_flat, active_weights,
            weights_flat, preserve_spectrum, self._carrier_seed(step_idx, frame_idx),
        )
        return idx_list, sign_list, combined

    def reconstruct(
        self,
        indices: List[int],
        signs: List[int],
        step_idx: int,
        frame_idx: int,
        schedule: ThreeBandStepSchedule,
        t: float,
        preserve_spectrum: bool = False,
    ) -> torch.Tensor:
        active_flat, active_weights, weights_flat = self._mask_weights(schedule, step_idx, t)
        D_band = int(active_flat.sum().item()) * self.C
        seed_sf = self._sf_seed(step_idx, frame_idx)
        signs_t = torch.tensor(signs, device=self.device, dtype=torch.float32)
        return self._regenerate_and_combine(
            seed_sf, indices, signs_t, D_band, active_flat, active_weights,
            weights_flat, preserve_spectrum, self._carrier_seed(step_idx, frame_idx),
        )

    def _regenerate_and_combine(
        self,
        seed_sf: int,
        indices: List[int],
        signs_t: torch.Tensor,
        D_band: int,
        active_flat: torch.Tensor,
        active_weights: torch.Tensor,
        weights_flat: torch.Tensor,
        preserve_spectrum: bool,
        carrier_seed: int,
    ) -> torch.Tensor:
        gen = torch.Generator(device=self.device).manual_seed(seed_sf)
        target = set(indices)
        found = {}
        for gi in range(self.K):
            atom = torch.randn(D_band, generator=gen, device=self.device)
            if gi in target:
                found[gi] = atom
            if len(found) >= len(target):
                break

        stacked = torch.stack([found[i] for i in indices])
        combined_vec = (signs_t.unsqueeze(1) * stacked).sum(0)
        return self._lift_to_spatial(
            combined_vec, active_flat, active_weights,
            weights_flat, preserve_spectrum, carrier_seed,
        )
