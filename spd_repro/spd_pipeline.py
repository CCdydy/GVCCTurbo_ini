"""
spd_pipeline.py — SPD inference pipeline for Wan 2.1-T2V (diffusers format).

Wraps the spectral noise expansion + timestep alignment + optimal schedule
components into a runnable progressive-resolution inference loop that
matches the SPD paper's Sec 4.1–4.2 algorithm.

Usage (programmatic):

    from spd_pipeline import SpectralProgressiveWan
    spd = SpectralProgressiveWan(
        model_dir="/path/to/Wan2.1-T2V-1.3B-Diffusers",
        delta=0.01, scales=[0.5, 1.0],
        beta=2.42, beta_intercept=0.0,  # from power_spectrum.py
    )
    spd.load("cuda", torch.bfloat16)
    frames = spd.generate(prompt="...", num_frames=33,
                          height=720, width=1280, num_steps=50)

The progressive schedule maps the 50 (or num_steps) baseline timesteps to
stages by transition times derived from Prop 2; each stage runs the
sub-segment of the schedule that falls within (t_i^*, t_{i-1}^*] at the
corresponding spatial resolution (s_i H, s_i W).

The pipeline reports its per-stage step count and wall-clock so a speedup
metric can be computed by run_spd_wan.py against a baseline.
"""

import os
import sys
import time
import math
from typing import List, Optional, Sequence, Tuple

import torch
import numpy as np
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from spectral_noise_expansion import spectral_noise_expand_aligned
from timestep_alignment import align
from optimal_schedule import (
    make_power_law_pofreq,
    build_progressive_schedule,
)


class SpectralProgressiveWan:
    """SPD inference for Wan 2.1-T2V (diffusers checkpoint format)."""

    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(
        self,
        model_dir: str,
        delta: float = 0.01,
        scales: Sequence[float] = (0.5, 1.0),
        beta: float = 2.42,
        beta_intercept: float = 0.0,
        flow_shift: float = 3.0,
    ):
        """
        Args:
            model_dir:      path to Wan 2.1-T2V-*-Diffusers checkpoint
            delta:          δ in (0, 1), SPD's single tolerance hyperparameter
            scales:         ascending list of resolution scales in (0, 1], last = 1.0
            beta:           power-law exponent β from power_spectrum.py fit
            beta_intercept: log-intercept c (so P_ω = exp(c) * |ω|^(-β))
            flow_shift:     scheduler flow_shift (Wan default 3.0 for 480p)
        """
        self.model_dir = os.path.abspath(model_dir)
        self.delta = delta
        self.scales = list(scales)
        assert abs(self.scales[-1] - 1.0) < 1e-9, "last scale must be 1.0"
        assert all(self.scales[i] < self.scales[i + 1] for i in range(len(self.scales) - 1))
        self.beta = beta
        self.beta_intercept = beta_intercept
        self.flow_shift = flow_shift

        self.device = None
        self.dtype = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from diffusers import WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler
        from transformers import UMT5EncoderModel, AutoTokenizer

        self.device = torch.device(device)
        self.dtype = dtype

        print(f"Loading Wan 2.1 T2V from {self.model_dir} (for SPD inference)")

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.model_dir, "tokenizer"))
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            os.path.join(self.model_dir, "text_encoder"),
            torch_dtype=dtype,
        ).eval().requires_grad_(False)
        self.vae = AutoencoderKLWan.from_pretrained(
            os.path.join(self.model_dir, "vae"),
            torch_dtype=torch.float32,
        ).eval().requires_grad_(False).to(self.device)
        self.transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(self.model_dir, "transformer"),
            torch_dtype=dtype,
        ).eval().requires_grad_(False).to(self.device)
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            self.model_dir, subfolder="scheduler"
        )

        # VAE latent normalization
        mean = torch.tensor(self.vae.config.latents_mean, dtype=torch.float32)
        std = torch.tensor(self.vae.config.latents_std, dtype=torch.float32)
        self.latent_mean = mean.view(1, -1, 1, 1, 1).to(self.device)
        self.latent_std = std.view(1, -1, 1, 1, 1).to(self.device)
        self.vae_stride = (4, 8, 8)
        self.latent_channels = self.vae.config.z_dim
        self._loaded = True
        print(f"  Loaded. delta={self.delta}, scales={self.scales}, β={self.beta}")

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(self, prompt: str, max_length: int = 226) -> torch.Tensor:
        self.text_encoder.to(self.device)
        inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = inputs.input_ids.to(self.device)
        mask = inputs.attention_mask.to(self.device)
        hidden = self.text_encoder(ids, attention_mask=mask).last_hidden_state.to(self.dtype)
        seq_len = mask.gt(0).sum(dim=1).long()
        trimmed = hidden[0, : seq_len[0]]
        padded = torch.cat([
            trimmed,
            trimmed.new_zeros(max_length - trimmed.size(0), trimmed.size(1)),
        ]).unsqueeze(0)
        return padded  # (1, L, D)

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def _power_law_func(self):
        return make_power_law_pofreq(beta=self.beta, intercept_log_c=self.beta_intercept)

    def _build_transition_times(self) -> List[float]:
        """Per Prop 2: t_i* for each transition s_i → s_{i+1}.

        Returns a list of length len(scales)-1 (descending in t).
        """
        sched = build_progressive_schedule(
            self.scales, self._power_law_func(), self.delta, omega_max_full=1.0)
        # sched: List[(s_i, s_next, t_i*)]
        return [t for _, _, t in sched]

    def _assign_timesteps_to_stages(
        self,
        baseline_timesteps: torch.Tensor,
        transition_times: List[float],
    ) -> List[torch.Tensor]:
        """Partition the baseline timesteps into stages by transition_times.

        Baseline timesteps are in [0, NUM_TRAIN_TIMESTEPS) (Wan native scale),
        descending from large t to 0.

        Stage i runs the subset where transition_times[i-1] <= t/1000 < transition_times[i-2]
        (with sentinel times 1.0 at the start and 0.0 at the end).

        Returns:
            List of N_stages tensors, each containing the subset of timesteps
            for that stage (in the original scale, descending).
        """
        S = len(self.scales)
        N_stages = S
        assert len(transition_times) == S - 1

        t_norm = baseline_timesteps.float().cpu() / float(self.NUM_TRAIN_TIMESTEPS)

        boundaries = [1.0] + transition_times + [0.0]  # length S+1
        stage_lists: List[torch.Tensor] = []
        for i in range(N_stages):
            hi = boundaries[i]
            lo = boundaries[i + 1]
            mask = (t_norm <= hi) & (t_norm > lo)
            # ensure the very first step (t=hi may equal 1.0) lands in stage 0
            if i == 0:
                mask = mask | (t_norm == 1.0)
            stage_lists.append(baseline_timesteps[mask])
        return stage_lists

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _stage_latent_shape(
        self, scale: float, num_frames: int, height: int, width: int
    ) -> Tuple[int, int, int, int]:
        """Latent shape (C, F_lat, H_lat, W_lat) at the requested scale."""
        f_lat = (num_frames - 1) // self.vae_stride[0] + 1
        h_full = height // self.vae_stride[1]
        w_full = width // self.vae_stride[2]
        h_lat = max(2, int(round(scale * h_full)))
        w_lat = max(2, int(round(scale * w_full)))
        # round to even for DiT patch_size=(1,2,2)
        h_lat += h_lat % 2
        w_lat += w_lat % 2
        return (self.latent_channels, f_lat, h_lat, w_lat)

    # ------------------------------------------------------------------
    # VAE
    # ------------------------------------------------------------------

    def _denorm(self, z_norm: torch.Tensor) -> torch.Tensor:
        return z_norm * self.latent_std + self.latent_mean

    @torch.no_grad()
    def decode_latent(self, z_norm: torch.Tensor) -> List[Image.Image]:
        if z_norm.dim() == 4:
            z_norm = z_norm.unsqueeze(0)
        raw_z = self._denorm(z_norm.float().to(self.device))
        video = self.vae.decode(raw_z, return_dict=False)[0]
        video = video.squeeze(0)
        video = (video / 2.0 + 0.5).clamp(0, 1)
        video = video.permute(1, 2, 3, 0).cpu().float().numpy()
        frames = []
        for f in range(video.shape[0]):
            img = (video[f] * 255).astype(np.uint8)
            frames.append(Image.fromarray(img))
        return frames

    # ------------------------------------------------------------------
    # Generation (progressive)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_frames: int = 33,
        height: int = 720,
        width: int = 1280,
        num_steps: int = 50,
        seed: int = 42,
        return_timings: bool = False,
    ):
        """SPD-accelerated text-to-video generation.

        Returns:
            frames: List[PIL.Image]
            (if return_timings) timing: dict with per-stage wall-clock + total
        """
        assert self._loaded, "call .load() first"
        prompt_embeds = self.encode_prompt(prompt)

        # Set up baseline scheduler timesteps with flow-matching shift
        self.scheduler.set_timesteps(
            num_inference_steps=num_steps,
            device=self.device,
            mu=self.flow_shift if hasattr(self.scheduler, "_mu") else None,
        )
        baseline_timesteps = self.scheduler.timesteps  # tensor, descending

        transition_times = self._build_transition_times()  # length S-1
        stage_timesteps = self._assign_timesteps_to_stages(baseline_timesteps, transition_times)

        # Initial scale and noise at s_1
        C, F_lat, H_lat_1, W_lat_1 = self._stage_latent_shape(self.scales[0], num_frames, height, width)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn(
            (1, C, F_lat, H_lat_1, W_lat_1),
            generator=gen, device=self.device, dtype=torch.float32,
        )

        timings = {"stages": [], "total_start": time.time(),
                   "schedule": [(s, n.numel()) for s, n in zip(self.scales, stage_timesteps)],
                   "transition_times": transition_times}

        # Iterate stages
        for stage_idx, (scale, stage_ts) in enumerate(zip(self.scales, stage_timesteps)):
            stage_t0 = time.time()
            steps_in_stage = stage_ts.numel()
            if steps_in_stage == 0:
                # No baseline timestep fell in this stage; just expand and continue
                if stage_idx < len(self.scales) - 1:
                    next_scale = self.scales[stage_idx + 1]
                    *_, H_next, W_next = self._stage_latent_shape(next_scale, num_frames, height, width)
                    r = next_scale / scale
                    # Use the stage's boundary t as the expansion time
                    t_at_expand = transition_times[stage_idx]
                    x_b, c, f, h, w = x.shape
                    x_flat = x.reshape(x_b * c * f, h, w)
                    x_exp, _ = spectral_noise_expand_aligned(
                        x_flat, H_next, W_next, t_i=t_at_expand, scale_ratio=r,
                        generator=torch.Generator(device=self.device).manual_seed(seed + 1000 + stage_idx),
                    )
                    x = x_exp.reshape(x_b, c, f, H_next, W_next)
                timings["stages"].append({"stage": stage_idx, "scale": scale,
                                          "steps": 0, "wall": time.time() - stage_t0})
                continue

            # Denoising loop at this resolution
            # (Use the scheduler's prediction step rather than rolling our own)
            for step_idx, t in enumerate(stage_ts):
                t_scalar = t.item()
                t_in = t.reshape(1).to(device=self.device, dtype=self.dtype)
                # Predict velocity
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    v = self.transformer(
                        hidden_states=x.to(self.dtype),
                        timestep=t_in,
                        encoder_hidden_states=prompt_embeds.to(self.dtype),
                        return_dict=False,
                    )[0]
                v = v.float()

                # Scheduler step (UniPC handles flow matching)
                x = self.scheduler.step(v, t, x, return_dict=False)[0]

            stage_wall = time.time() - stage_t0
            timings["stages"].append({"stage": stage_idx, "scale": scale,
                                      "steps": steps_in_stage, "wall": stage_wall,
                                      "lat_hw": tuple(x.shape[-2:])})

            # Inter-stage expansion (if not last)
            if stage_idx < len(self.scales) - 1:
                next_scale = self.scales[stage_idx + 1]
                *_, H_next, W_next = self._stage_latent_shape(next_scale, num_frames, height, width)
                r = next_scale / scale
                # SPD expansion happens at the transition time t_i* (= boundaries[i+1])
                # which is the t value used to scale the high-freq noise injection
                t_at_expand = transition_times[stage_idx]
                x_b, c, f, h, w = x.shape
                x_flat = x.reshape(x_b * c * f, h, w)
                x_exp, t_aligned = spectral_noise_expand_aligned(
                    x_flat, H_next, W_next, t_i=t_at_expand, scale_ratio=r,
                    generator=torch.Generator(device=self.device).manual_seed(seed + 1000 + stage_idx),
                )
                x = x_exp.reshape(x_b, c, f, H_next, W_next)
                # Reset UniPC multistep history: cached previous velocities are at
                # the old resolution and would dimension-mismatch on next step.
                # First step of the new stage becomes first-order (Euler-like),
                # then the buffer rebuilds as denoising continues.
                self.scheduler.model_outputs = [None] * self.scheduler.config.solver_order
                self.scheduler.lower_order_nums = 0
                self.scheduler.last_sample = None

        # Reset scheduler for next call (UniPC keeps state)
        self.scheduler.set_timesteps(num_inference_steps=num_steps, device=self.device)

        timings["total_wall"] = time.time() - timings["total_start"]
        frames = self.decode_latent(x)

        if return_timings:
            return frames, timings
        return frames


# ============================================================================
# Smoke test (no actual generation; just construct + schedule)
# ============================================================================

def _smoke_test():
    """Verify the schedule + stage assignment logic without loading the model."""
    spd = SpectralProgressiveWan(
        model_dir="/dev/null",  # not loaded
        delta=0.01, scales=[0.5, 0.75, 1.0],
        beta=2.42, beta_intercept=0.0,
    )
    transition_times = spd._build_transition_times()
    print(f"Transition times for scales={spd.scales}, β={spd.beta}, δ={spd.delta}:")
    for i, t in enumerate(transition_times):
        print(f"  scale {spd.scales[i]} → {spd.scales[i+1]} at t = {t:.4f}")

    # Mock baseline timesteps (Wan UniPC, descending from 999 to 0 over 50 steps)
    baseline = torch.linspace(999, 0, 50).long()
    stages = spd._assign_timesteps_to_stages(baseline, transition_times)
    print(f"\nStage assignment with 50 baseline steps:")
    for i, st in enumerate(stages):
        print(f"  stage {i} (scale {spd.scales[i]}): {st.numel()} steps, range t ∈ [{st.min().item() if st.numel() else '—'}, {st.max().item() if st.numel() else '—'}]")


if __name__ == "__main__":
    _smoke_test()
