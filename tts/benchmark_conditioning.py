#!/usr/bin/env python3
"""Measure where Chatterbox's time actually goes: model load, reference-clip
conditioning (stage 1), and per-line generation given already-prepared
conditioning (stage 2). Used to decide whether caching conditioning is
worth it, and to quantify the win, before touching tts_cli.py's synthesis
loop.
"""
import time

from chatterbox_device import load_chatterbox_model

TEST_LINE = (
    "Carbon can bond to itself over and over, forming chains and rings, "
    "and that's basically why organic chemistry exists at all."
)
REF_CLIP = "reference_voices/candidates/p311_male.wav"


def main():
    t0 = time.perf_counter()
    model, device = load_chatterbox_model()
    t1 = time.perf_counter()
    load_time = t1 - t0
    print(f"\n[1] Model load time ({device}): {load_time:.2f}s\n")

    # Stage 1 in isolation: prepare_conditionals() alone, no generation.
    t0 = time.perf_counter()
    model.prepare_conditionals(REF_CLIP)
    t1 = time.perf_counter()
    prep_time = t1 - t0
    print(f"[2] prepare_conditionals() alone (stage 1, run once): {prep_time:.3f}s")

    cached_conds = model.conds  # this is what we'd cache

    # Stage 2 in isolation: generate(text) reusing cached conds, no
    # audio_prompt_path -> skips prepare_conditionals entirely.
    stage2_times = []
    for i in range(3):
        model.conds = cached_conds  # simulate swapping in cached conditioning
        t0 = time.perf_counter()
        model.generate(TEST_LINE)
        t1 = time.perf_counter()
        stage2_times.append(t1 - t0)
        print(f"[3.{i}] generate(text) alone, cached conds (stage 2): {stage2_times[-1]:.3f}s")

    avg_stage2 = sum(stage2_times) / len(stage2_times)

    # Combined, uncached: generate(text, audio_prompt_path=...) - the OLD
    # behavior - redoes prepare_conditionals() every single call.
    combined_times = []
    for i in range(3):
        t0 = time.perf_counter()
        model.generate(TEST_LINE, audio_prompt_path=REF_CLIP)
        t1 = time.perf_counter()
        combined_times.append(t1 - t0)
        print(f"[4.{i}] generate(text, audio_prompt_path=...) uncached (stage 1+2): {combined_times[-1]:.3f}s")

    avg_combined = sum(combined_times) / len(combined_times)

    print("\n--- Summary ---")
    print(f"Model load:                          {load_time:.2f}s (one-time per process)")
    print(f"Stage 1 (prepare_conditionals) alone: {prep_time:.3f}s")
    print(f"Stage 2 (generate, cached) avg:        {avg_stage2:.3f}s")
    print(f"Stage 1+2 combined (uncached) avg:     {avg_combined:.3f}s")
    print(f"Implied stage-1 overhead per call:     {avg_combined - avg_stage2:.3f}s")
    print(f"Stage 1 as % of combined call time:    {100 * (avg_combined - avg_stage2) / avg_combined:.1f}%")


if __name__ == "__main__":
    main()
