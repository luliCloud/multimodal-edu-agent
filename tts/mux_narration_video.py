#!/usr/bin/env python3
"""Mux a narration track onto a video, producing one file whose embedded
video and audio play together automatically - no separate audio file, no
manual syncing by the viewer.

Works for any video + narration WAV that were timed against each other:
a pre-existing reference video (like little-seed.MP4, scripted by hand
against its visual beats) or a video rendered by backend/'s mock/wan/
cuda_probe generators via sync_storyboard_audio.py.

Usage:
    python mux_narration_video.py little-seed.MP4 little_seed_narration.wav \
        --output little_seed_final.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: <video stem>_final.mp4 beside the video")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not args.audio.exists():
        print(f"ERROR: audio not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.video.with_name(args.video.stem + "_final.mp4")

    video_len = probe_duration(args.video)
    audio_len = probe_duration(args.audio)
    if audio_len > video_len + 0.25:
        print(
            f"WARNING: narration ({audio_len:.2f}s) is longer than the video "
            f"({video_len:.2f}s) - the last {audio_len - video_len:.2f}s of "
            "narration will be cut off, since the video has no more frames "
            "to show. Trim the script or extend the video to fix this.",
            file=sys.stderr,
        )

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(args.video), "-i", str(args.audio),
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            str(output),
        ],
        capture_output=True,
    )
    if result.returncode:
        print(result.stderr.decode("utf-8", errors="replace")[-1000:], file=sys.stderr)
        sys.exit(1)

    print(f"video:  {args.video} ({video_len:.2f}s)")
    print(f"audio:  {args.audio} ({audio_len:.2f}s)")
    print(f"-> {output} ({probe_duration(output):.2f}s), both streams in one file")


if __name__ == "__main__":
    main()
