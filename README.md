# xdsp_domain

MODULE NAME:
**xc_domain**

DESCRIPTION:
Sample-rate / time-domain core for XDSP.
Generates absolute time in seconds and maintains a rolling sample counter, with smoothed, time-varying sample-rate support. Provides per-sample Δt, radians-per-sample, and unit-conversion scalars in a purely functional, Numba-JIT-friendly style. Designed as a “domain generator” module that other DSP modules (oscillators, envelopes, LFOs, schedulers) can depend on.

INPUTS:

* `target_sr` : Desired (possibly time-varying) sample rate in Hz (scalar or per-sample array in process).
* `alpha` : Sample-rate smoothing coefficient (0 ≤ alpha < 1). Higher = slower changes.
* `n_samples` : Block length for `*_process` (number of time steps).

OUTPUTS:

* `t[n]` : Absolute time in seconds at each sample (vector from `*_process`, scalar from `*_tick`).
* (implicitly available via state)

  * `k` : Global sample index.
  * `sr_s` : Smoothed sample rate.
  * `spu` : Seconds per sample (`1 / sr_s`).
  * `rad_per_sample` : Radians-per-sample factor (`2π * spu`).

STATE VARIABLES (tuple):
`state = (sample_rate, ups, spu, rad_per_sample, sample_counter, time_seconds, smooth_sr)`

* `sample_rate` : Current smoothed sample rate (Hz).
* `ups` : Seconds→samples scalar (`sample_rate`).
* `spu` : Samples→seconds scalar (`1 / sample_rate`).
* `rad_per_sample` : Radians-per-sample scalar (`2π * spu`).
* `sample_counter` : Global sample index (integer, monotonically increasing).
* `time_seconds` : Absolute time in seconds for the *last* generated sample.
* `smooth_sr` : Internal one-pole smoothed sample-rate state.

EQUATIONS / MATH:

Sample-rate smoothing (one-pole lowpass in time):

* Let `sr_target[n]` be the desired sample rate (constant or time-varying).
* Let `α` (`alpha`) be the smoothing coefficient.

One-pole smoothing:

* `sr_s[n] = α * sr_s[n-1] + (1 - α) * sr_target[n]`

Instantaneous step duration:

* `Δt[n] = 1 / sr_s[n]`

Absolute time:

* `t[n] = t[n-1] + Δt[n]`

Global sample index:

* `k[n] = k[n-1] + 1`

Unit scalars at sample `n`:

* `ups[n] = sr_s[n]`  (seconds → samples)
* `spu[n] = Δt[n] = 1 / sr_s[n]`  (samples → seconds)
* `rad_per_sample[n] = 2π * spu[n]`

Tick output:

* `y[n] = t[n]`  (scalar time for this sample)

State update (from `n` to `n+1`):

* `sample_rate[n+1] = sr_s[n]`
* `ups[n+1] = ups[n]` updated from `sr_s[n]`
* `spu[n+1] = spu[n]` updated from `sr_s[n]`
* `rad_per_sample[n+1] = 2π * spu[n+1]`
* `sample_counter[n+1] = k[n]`
* `time_seconds[n+1] = t[n]`
* `smooth_sr[n+1] = sr_s[n]`

through-zero rules:

* Not applicable (no oscillator phase here). Sample counter only increments (no wrap, no through-zero).

phase wrapping rules:

* Not handled here. `rad_per_sample` provides the increment factor; oscillators perform their own phase accumulation and wrapping (`phase = (phase + inc) % (2π)` or equivalent).

nonlinearities:

* Only the `1 / sr_s` reciprocal, assumed stable for `sr_s > 0`.

interpolation rules:

* Temporal interpolation of sample rate is via the one-pole smoother `sr_s[n]`.
* No explicit interpolation of time; `t[n]` is exact sum of variable step sizes.

time-varying coefficient rules:

* `sr_target[n]` can vary per sample (via an input array to `*_process`).
* `sr_s[n]` follows `sr_target[n]` according to the smoothing coefficient `α`.

NOTES:

* Must maintain `sr_target[n] > 0` to avoid division by zero; caller responsible for clamping.
* For typical use, `α` ∈ [0, 0.9999]. `α = 0` means no smoothing (instant update).
* This module does *not* do block scheduling or callbacks; it just generates domain scalars. Higher-level graph/domain-routing is built on top.
* All state is passed as tuples; no classes, dicts, or side effects.
* All Numba functions are decorated with `@njit(cache=True, fastmath=True)`.

---

## FULL PYTHON FILE: `xc_domain.py`

```python
"""
xc_domain.py

XC-DSP core: sample-rate / time-domain generator in pure functional XDSP style.

This module provides a domain generator that maintains:

- A smoothed, possibly time-varying sample rate (Hz).
- A global sample counter k (integer).
- Absolute time t in seconds.
- Unit scalars:
    ups = sample_rate          (seconds -> samples)
    spu = 1 / sample_rate      (samples -> seconds)
    rad_per_sample = 2π * spu  (radians per sample)

The core recurrence per sample n:

    sr_s[n]      = alpha * sr_s[n-1] + (1 - alpha) * sr_target[n]
    Δt[n]        = 1 / sr_s[n]
    t[n]         = t[n-1] + Δt[n]
    k[n]         = k[n-1] + 1
    rad_ps[n]    = 2π * Δt[n]

State is carried as a tuple:

    state = (sample_rate, ups, spu, rad_per_sample,
             sample_counter, time_seconds, smooth_sr)

There are no classes or dicts; everything is functional and Numba-friendly.

Public functions (XDSP style):

    xc_domain_init(...)
    xc_domain_update_state(...)
    xc_domain_tick(target_sr, alpha, state)
    xc_domain_process(n_samples, target_sr, alpha, state)

- *_tick returns (y, new_state), where y is scalar time in seconds.
- *_process returns (t, new_state), where t is an array of times, length n_samples.

All heavy per-sample work is done in @njit(cache=True, fastmath=True) functions.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Constants and state indexing
# ---------------------------------------------------------------------------

TWO_PI = 2.0 * math.pi

# State tuple layout:
# (sample_rate, ups, spu, rad_per_sample, sample_counter, time_seconds, smooth_sr)
STATE_SAMPLE_RATE = 0
STATE_UPS = 1
STATE_SPU = 2
STATE_RAD_PER_SAMPLE = 3
STATE_SAMPLE_COUNTER = 4
STATE_TIME_SECONDS = 5
STATE_SMOOTH_SR = 6

DEFAULT_SR = 48_000.0


# ---------------------------------------------------------------------------
# Initialization and state update
# ---------------------------------------------------------------------------

def xc_domain_init(
    sample_rate: float = DEFAULT_SR,
    initial_sample_counter: int = 0,
) -> Tuple[float, float, float, float, int, float, float]:
    """
    Initialize domain state.

    Parameters
    ----------
    sample_rate : float
        Initial sample rate in Hz.
    initial_sample_counter : int
        Starting global sample index k0.

    Returns
    -------
    state : tuple
        (sample_rate, ups, spu, rad_per_sample,
         sample_counter, time_seconds, smooth_sr)
    """
    sr = float(sample_rate)
    k0 = int(initial_sample_counter)
    ups = sr
    spu = 1.0 / sr
    rad_per_sample = TWO_PI * spu
    time_seconds = k0 * spu
    smooth_sr = sr

    state = (
        sr,             # sample_rate
        ups,            # ups
        spu,            # spu
        rad_per_sample, # rad_per_sample
        k0,             # sample_counter
        time_seconds,   # time_seconds
        smooth_sr,      # smooth_sr
    )
    return state


@njit(cache=True, fastmath=True)
def _xc_domain_recalc_state_jit(
    sample_rate_new: float,
    sample_counter: int,
) -> Tuple[float, float, float, float, int, float, float]:
    """
    Jitted helper: recompute all derived scalars from a new sample rate and
    an existing sample counter.

    This is used by xc_domain_update_state().
    """
    ups = sample_rate_new
    spu = 1.0 / sample_rate_new
    rad_per_sample = TWO_PI * spu
    time_seconds = sample_counter * spu
    smooth_sr = sample_rate_new

    return (
        sample_rate_new,
        ups,
        spu,
        rad_per_sample,
        sample_counter,
        time_seconds,
        smooth_sr,
    )


def xc_domain_update_state(
    sample_rate_new: float,
    state: Tuple[float, float, float, float, int, float, float],
) -> Tuple[float, float, float, float, int, float, float]:
    """
    Hard-update the domain's sample rate without temporal smoothing.

    This is intended for configuration changes between processing blocks,
    not per-sample changes.

    Parameters
    ----------
    sample_rate_new : float
        New sample rate in Hz (must be > 0).
    state : tuple
        Existing domain state.

    Returns
    -------
    new_state : tuple
        Updated state with recalculated ups, spu, rad_per_sample and
        time_seconds, preserving sample_counter.
    """
    sample_counter = int(state[STATE_SAMPLE_COUNTER])
    return _xc_domain_recalc_state_jit(float(sample_rate_new), sample_counter)


# ---------------------------------------------------------------------------
# Tick: single-sample update
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def xc_domain_tick_jit(
    target_sr: float,
    alpha: float,
    state: Tuple[float, float, float, float, int, float, float],
) -> Tuple[float, Tuple[float, float, float, float, int, float, float]]:
    """
    Jitted single-sample domain update.

    Parameters
    ----------
    target_sr : float
        Desired sample rate for this sample.
    alpha : float
        Smoothing coefficient in [0, 1). alpha=0 => no smoothing.
    state : tuple
        Domain state.

    Returns
    -------
    y : float
        Time in seconds for this sample (after update).
    new_state : tuple
        Updated state.
    """
    sample_rate = state[STATE_SAMPLE_RATE]
    ups = state[STATE_UPS]
    spu = state[STATE_SPU]
    rad_per_sample = state[STATE_RAD_PER_SAMPLE]
    sample_counter = state[STATE_SAMPLE_COUNTER]
    time_seconds = state[STATE_TIME_SECONDS]
    smooth_sr = state[STATE_SMOOTH_SR]

    # One-pole smoothing of sample rate
    one_minus_alpha = 1.0 - alpha
    sr_s = alpha * smooth_sr + one_minus_alpha * target_sr
    if sr_s <= 0.0:
        # Very basic safety clamp; in practice caller should ensure positivity.
        sr_s = 1e-9

    # Recompute unit scalars
    ups = sr_s
    spu = 1.0 / sr_s
    rad_per_sample = TWO_PI * spu

    # Advance sample counter and time
    sample_counter = sample_counter + 1
    time_seconds = time_seconds + spu

    # New state
    sample_rate = sr_s
    smooth_sr = sr_s

    new_state = (
        sample_rate,
        ups,
        spu,
        rad_per_sample,
        sample_counter,
        time_seconds,
        smooth_sr,
    )

    # Output is absolute time in seconds for this sample
    y = time_seconds
    return y, new_state


def xc_domain_tick(
    target_sr: float,
    alpha: float,
    state: Tuple[float, float, float, float, int, float, float],
) -> Tuple[float, Tuple[float, float, float, float, int, float, float]]:
    """
    Public wrapper for single-sample domain update.

    Returns (y, new_state), where y is time in seconds.
    """
    return xc_domain_tick_jit(float(target_sr), float(alpha), state)


# ---------------------------------------------------------------------------
# Process: block update
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _xc_domain_process_jit(
    t_out: np.ndarray,
    target_sr_array: np.ndarray,
    alpha: float,
    state: Tuple[float, float, float, float, int, float, float],
) -> Tuple[float, float, float, float, int, float, float]:
    """
    Jitted block processor.

    Parameters
    ----------
    t_out : np.ndarray (float64)
        Preallocated output array for time in seconds; shape (n_samples,).
    target_sr_array : np.ndarray (float64)
        Per-sample target sample rate in Hz; shape (n_samples,).
    alpha : float
        Smoothing coefficient in [0, 1).
    state : tuple
        Initial domain state.

    Returns
    -------
    new_state : tuple
        State after processing the entire block.
    """
    sample_rate = state[STATE_SAMPLE_RATE]
    ups = state[STATE_UPS]
    spu = state[STATE_SPU]
    rad_per_sample = state[STATE_RAD_PER_SAMPLE]
    sample_counter = state[STATE_SAMPLE_COUNTER]
    time_seconds = state[STATE_TIME_SECONDS]
    smooth_sr = state[STATE_SMOOTH_SR]

    one_minus_alpha = 1.0 - alpha
    n = t_out.shape[0]

    for i in range(n):
        target_sr = target_sr_array[i]

        # One-pole smoothing of sample rate
        sr_s = alpha * smooth_sr + one_minus_alpha * target_sr
        if sr_s <= 0.0:
            sr_s = 1e-9

        # Recompute unit scalars
        ups = sr_s
        spu = 1.0 / sr_s
        rad_per_sample = TWO_PI * spu

        # Advance sample counter and time
        sample_counter = sample_counter + 1
        time_seconds = time_seconds + spu

        # Store for next iteration
        sample_rate = sr_s
        smooth_sr = sr_s

        # Output time for this sample
        t_out[i] = time_seconds

    new_state = (
        sample_rate,
        ups,
        spu,
        rad_per_sample,
        sample_counter,
        time_seconds,
        smooth_sr,
    )
    return new_state


def xc_domain_process(
    n_samples: int,
    target_sr,
    alpha: float,
    state: Tuple[float, float, float, float, int, float, float],
) -> Tuple[np.ndarray, Tuple[float, float, float, float, int, float, float]]:
    """
    Block-based domain update.

    Parameters
    ----------
    n_samples : int
        Number of samples in this block.
    target_sr : float or array-like
        Desired sample rate(s) for this block. If scalar, it is broadcast.
        If array, it must have length n_samples.
    alpha : float
        Smoothing coefficient in [0, 1).
    state : tuple
        Initial domain state.

    Returns
    -------
    t : np.ndarray
        Time in seconds for each sample; shape (n_samples,).
    new_state : tuple
        State after processing the block.
    """
    n = int(n_samples)
    # Preallocate outputs (no allocation inside jitted code)
    t_out = np.empty(n, dtype=np.float64)

    # Normalize target_sr to an array
    target_sr_arr = np.asarray(target_sr, dtype=np.float64)
    if target_sr_arr.ndim == 0:
        # Broadcast scalar to full block
        target_sr_arr = np.full(n, float(target_sr_arr), dtype=np.float64)
    else:
        if target_sr_arr.shape[0] != n:
            raise ValueError("target_sr array must have length n_samples")

    new_state = _xc_domain_process_jit(t_out, target_sr_arr, float(alpha), state)
    return t_out, new_state


# ---------------------------------------------------------------------------
# Unit conversion helpers (vectorized)
# ---------------------------------------------------------------------------

def xc_domain_seconds_to_samples(
    seconds,
    state: Tuple[float, float, float, float, int, float, float],
) -> np.ndarray:
    """
    Vectorized seconds -> samples using current ups from state.
    """
    ups = float(state[STATE_UPS])
    return np.asarray(seconds, dtype=np.float64) * ups


def xc_domain_samples_to_seconds(
    samples,
    state: Tuple[float, float, float, float, int, float, float],
) -> np.ndarray:
    """
    Vectorized samples -> seconds using current spu from state.
    """
    spu = float(state[STATE_SPU])
    return np.asarray(samples, dtype=np.float64) * spu


def xc_domain_hz_to_phase_inc(
    hz,
    state: Tuple[float, float, float, float, int, float, float],
) -> np.ndarray:
    """
    Vectorized Hz -> radians-per-sample using current spu from state.
    """
    spu = float(state[STATE_SPU])
    return np.asarray(hz, dtype=np.float64) * (TWO_PI * spu)


# ---------------------------------------------------------------------------
# Smoke test & examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple smoke test and examples:
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    # --- Smoke test: constant SR -----------------------------------------
    sr = 48_000.0
    state = xc_domain_init(sample_rate=sr)
    n = 1024
    alpha = 0.0  # no smoothing

    t, state = xc_domain_process(n_samples=n, target_sr=sr, alpha=alpha, state=state)

    print("Final sample_counter:", state[STATE_SAMPLE_COUNTER])
    print("Final time_seconds:", state[STATE_TIME_SECONDS])
    print("Approx block duration (s):", n / sr)

    # --- Plot example: SR ramp -------------------------------------------
    n_plot = 5000
    sr_start = 44_100.0
    sr_end = 96_000.0
    alpha_plot = 0.99

    state_plot = xc_domain_init(sample_rate=sr_start)

    # Linear ramp of target SR for visualization
    target_sr_ramp = np.linspace(sr_start, sr_end, n_plot, dtype=np.float64)
    t_ramp, state_plot = xc_domain_process(
        n_samples=n_plot,
        target_sr=target_sr_ramp,
        alpha=alpha_plot,
        state=state_plot,
    )

    # Reconstruct smoothed SR from state transitions (approx) for plotting
    # Here, we just recompute from instantaneous Δt (spu) as 1 / derivative of time.
    dt = np.diff(t_ramp, prepend=t_ramp[0])
    sr_est = 1.0 / np.maximum(dt, 1e-9)

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t_ramp, target_sr_ramp, label="target SR (Hz)", alpha=0.5)
    plt.plot(t_ramp, sr_est, label="smoothed SR estimate (Hz)")
    plt.xlabel("time (s)")
    plt.ylabel("sample rate (Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(t_ramp, t_ramp, label="t[n]", color="C1")
    plt.xlabel("time (s)")
    plt.ylabel("t[n] (s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Listen example: simple sine using domain ------------------------
    if HAVE_SD:
        duration = 1.0  # seconds
        n_audio = int(sr * duration)
        state_audio = xc_domain_init(sample_rate=sr)

        # Constant SR; alpha=0 (no smoothing)
        t_audio, state_audio = xc_domain_process(
            n_samples=n_audio,
            target_sr=sr,
            alpha=0.0,
            state=state_audio,
        )

        freq = 440.0
        phase = 2.0 * math.pi * freq * t_audio
        y = 0.2 * np.sin(phase).astype(np.float32)

        print("Playing 440 Hz test tone...")
        sd.play(y, samplerate=int(sr))
        sd.wait()
    else:
        print("sounddevice not available; skipping audio example.")
```

---

### Next things to do

Here are some natural follow-ons that plug into this `xc_domain` core:

1. **`xc_osc_sine` module**

   * Uses `xc_domain_hz_to_phase_inc` and `t[n]` to generate phase and audio.
   * Adds through-zero phase wrap and frequency modulation.

2. **`xc_env_adsr` module**

   * Time-based envelope using `t[n]` and/or sample counter, with vectorized shape and parameter smoothing.

3. **`xc_scheduler` / event domain module**

   * Converts event times (seconds, beats, samples) into sample indices using this domain’s `ups` / `spu`.

Whenever you’re ready, say something like “Generate the sine oscillator module next” and I’ll build the next XDSP-style file on top of this domain.
