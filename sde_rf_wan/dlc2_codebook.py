"""
dlc2_codebook.py — DLC2 structured high-freq codebook for V1 transitions.

Implements Version A from RESEARCH_NOTES.md "DLC2: AsymFlow-Inspired Structured
Innovation": atoms live in a low-rank subspace of the DCT high-frequency band.

For each transition s_i → s_{i+1} with latent shape change (H_src, W_src) →
(H_tgt, W_tgt):

  High-freq region:    HF = {(h, w) : h ≥ H_src OR w ≥ W_src}
                       |HF| = H_tgt·W_tgt - H_src·W_src
                       D_hi = C · |HF|

  Sub-band selection:  rank r ≤ D_hi
                       basis = top-r HF positions sorted by frequency
                       magnitude ||ω||₁ (lowest first → closer to band
                       cutoff where residual power concentrates per
                       spectral power law)

  Atoms:               q_k ~ N(0, I_r), placed only in selected r DCT
                       positions (zeros elsewhere). iDCT to spatial gives
                       a structured high-freq-only atom.

The codebook stores no atom data — atoms are deterministically regenerated
from (step, frame, K_idx) seed, like TurboPerFrameCodebook.
"""

import math
from typing import List, Tuple

import torch


class DLC2HighFreqCodebook:
    """Sub-band low-rank codebook on the DCT high-freq region.

    Parameters
    ----------
    K : codebook size (same as cb_full in dlc1_codebook for fair comparison)
    M : atoms selected per frame per transition
    src_lat_shape : (C, H_src, W_src) — latent at source stage
    tgt_lat_shape : (C, H_tgt, W_tgt) — latent at target stage
    rank_frac : fraction of D_hi used as the sub-band (0 < rank_frac ≤ 1)
    seed : base RNG seed (combined with step/frame indices)
    device : torch device
    """

    def __init__(
        self,
        K: int,
        M: int,
        src_lat_shape: Tuple[int, int, int],
        tgt_lat_shape: Tuple[int, int, int],
        rank_frac: float = 0.5,
        seed: int = 42,
        device: str = "cuda",
    ):
        self.K = K
        self.M = M
        self.C, self.H_src, self.W_src = src_lat_shape
        _, self.H_tgt, self.W_tgt = tgt_lat_shape
        self.rank_frac = rank_frac
        self.seed = seed
        self.device = device

        # Build HF mask and rank-r position list
        mask_lo = torch.zeros(self.H_tgt, self.W_tgt, dtype=torch.bool)
        mask_lo[:self.H_src, :self.W_src] = True
        self.mask_hi = (~mask_lo).to(device)  # (H_tgt, W_tgt)

        # High-freq positions sorted by frequency magnitude (||(h, w)||)
        hf_coords = torch.nonzero(~mask_lo, as_tuple=False).float()  # (n_hi, 2)
        # Frequency magnitude per DCT convention (just (h, w) Euclid distance from origin)
        freq_mag = torch.sqrt(hf_coords[:, 0]**2 + hf_coords[:, 1]**2)
        sort_idx = freq_mag.argsort()  # ascending = closer to band cutoff first
        hf_coords_sorted = hf_coords[sort_idx].long()  # (n_hi, 2)

        n_hi_total = hf_coords_sorted.shape[0]
        n_hi_rank = max(1, int(round(n_hi_total * rank_frac)))
        self.n_hi = n_hi_rank
        self.D_hi = self.C * n_hi_rank

        # Sub-band positions (the ones we actually use): shape (n_hi_rank, 2)
        self.subband_coords = hf_coords_sorted[:n_hi_rank].to(device)

        # Build linear-indexable mask for the sub-band: flat (H_tgt*W_tgt,) bool
        flat = torch.zeros(self.H_tgt * self.W_tgt, dtype=torch.bool, device=device)
        flat_idx = self.subband_coords[:, 0] * self.W_tgt + self.subband_coords[:, 1]
        flat[flat_idx] = True
        self.subband_mask_flat = flat   # (H_tgt*W_tgt,)
        self.subband_mask_hw = flat.view(self.H_tgt, self.W_tgt)  # (H_tgt, W_tgt)

        self.bits_per_index = math.ceil(math.log2(K)) if K > 1 else 1
        self.gen_batch = max(32, min(512, 4096 // max(1, self.D_hi // 1024)))

    def _seed_for(self, step: int, frame: int) -> int:
        return self.seed * 100003 + step * 10007 + frame * 31

    def _project_to_subband(self, x_dct: torch.Tensor) -> torch.Tensor:
        """Extract sub-band coefficients from DCT tensor.

        x_dct: (C, H_tgt, W_tgt) DCT coefficients
        Returns: (D_hi,) flat vector of sub-band coefficients
        """
        flat = x_dct.reshape(self.C, -1)  # (C, H_tgt*W_tgt)
        return flat[:, self.subband_mask_flat].reshape(-1)

    def _lift_to_dct(self, vec: torch.Tensor) -> torch.Tensor:
        """Inverse of _project_to_subband: place coefficients back in shape.

        vec: (D_hi,) flat vector
        Returns: (C, H_tgt, W_tgt) DCT with values only in sub-band
        """
        out_flat = torch.zeros(self.C, self.H_tgt * self.W_tgt, device=self.device)
        out_flat[:, self.subband_mask_flat] = vec.reshape(self.C, self.n_hi)
        return out_flat.reshape(self.C, self.H_tgt, self.W_tgt)

    def _generate_atoms_batch(
        self, gen: torch.Generator, count: int
    ) -> torch.Tensor:
        """Generate `count` atoms in sub-band as (count, D_hi) tensor."""
        atoms = torch.empty(count, self.D_hi, device=self.device)
        for i in range(count):
            atoms[i] = torch.randn(self.D_hi, generator=gen, device=self.device)
        return atoms

    def select_atoms(
        self,
        residual_dct: torch.Tensor,
        step_idx: int,
        frame_idx: int,
    ) -> Tuple[List[int], List[int], torch.Tensor]:
        """Select top-M atoms by |⟨atom, projected_residual⟩|.

        Parameters
        ----------
        residual_dct : (C, H_tgt, W_tgt) — DCT of residual (low-freq band
                       may be ignored; only sub-band positions are used).
        step_idx : transition step index
        frame_idx : temporal frame index

        Returns
        -------
        indices : list of M atom indices in [0, K)
        signs   : list of M signs (±1)
        combined : (C, H_tgt, W_tgt) iDCT of summed selected atoms (lives
                   entirely in the sub-band by construction).
        """
        r_vec = self._project_to_subband(residual_dct)
        seed_sf = self._seed_for(step_idx, frame_idx)

        gen = torch.Generator(device=self.device).manual_seed(seed_sf)
        all_ips = torch.empty(self.K, device=self.device)
        for b0 in range(0, self.K, self.gen_batch):
            bc = min(self.gen_batch, self.K - b0)
            atoms = self._generate_atoms_batch(gen, bc)
            all_ips[b0:b0 + bc] = atoms @ r_vec
            del atoms

        _, top_idx = all_ips.abs().topk(self.M)
        top_idx_sorted = top_idx.sort().values
        signs_t = torch.sign(all_ips[top_idx_sorted])
        signs_t[signs_t == 0] = 1.0

        idx_list = top_idx_sorted.cpu().tolist()
        sign_list = signs_t.cpu().int().tolist()

        combined_dct = self._regenerate_and_combine(seed_sf, idx_list, signs_t)
        return idx_list, sign_list, combined_dct

    def reconstruct(
        self,
        indices: List[int],
        signs: List[int],
        step_idx: int,
        frame_idx: int,
    ) -> torch.Tensor:
        """Decoder: reconstruct (C, H_tgt, W_tgt) DCT-domain combined atom."""
        seed_sf = self._seed_for(step_idx, frame_idx)
        signs_t = torch.tensor(signs, device=self.device, dtype=torch.float32)
        return self._regenerate_and_combine(seed_sf, indices, signs_t)

    def _regenerate_and_combine(
        self,
        seed_sf: int,
        indices: List[int],
        signs_t: torch.Tensor,
    ) -> torch.Tensor:
        """Regenerate selected atoms, combine with signs, normalize, place in DCT."""
        gen = torch.Generator(device=self.device).manual_seed(seed_sf)
        target = set(indices)
        found = {}
        for gi in range(self.K):
            atom = torch.randn(self.D_hi, generator=gen, device=self.device)
            if gi in target:
                found[gi] = atom
            if len(found) == len(target):
                break

        # Stack in index order and combine
        stacked = torch.stack([found[i] for i in indices])  # (M, D_hi)
        combined_vec = (signs_t.unsqueeze(1) * stacked).sum(0)  # (D_hi,)

        # Two-step normalization:
        # (1) unit-variance in sub-band coefficients (matches DLC1 unit-variance
        #     convention at the relevant subspace level)
        # (2) scale by sqrt(1/rank_frac) so that after iDCT the spatial signal
        #     has std ≈ sqrt(D_hi_full/D) — matching DLC1's high-freq portion
        #     spatial magnitude. Without (2), sub-band concentration produces
        #     a strictly smaller prior contribution than DLC1, conflating the
        #     "subspace restriction" question with a magnitude artifact.
        std = combined_vec.std()
        if std > 1e-8:
            combined_vec = combined_vec / std
        combined_vec = combined_vec * math.sqrt(1.0 / self.rank_frac)

        return self._lift_to_dct(combined_vec)
