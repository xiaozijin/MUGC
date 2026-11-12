# -*- coding: utf-8 -*-
"""Frame selection and extraction for the VG Test Caption pipeline.

Frames are sampled at t = 0, 5, 10, ... while t < floor(duration),
and each frame is exported as a JPEG at quality -q:v 2, at the original
resolution (no resize / crop / padding).
"""
import argparse
import os
import subprocess
import tempfile

DEFAULT_INTERVAL = 5


def get_duration(path):
    """Return the clip duration in seconds (float)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, timeout=60)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def frame_times(duration, interval=DEFAULT_INTERVAL):
    """Timestamps sampled from the clip.

    Matches the released pipeline exactly:
        times = list(range(0, int(duration), interval))
    so a 5.04 s clip yields [0].
    """
    return list(range(0, int(duration), interval))


def extract_frame(video_path, t, out_path, quality=2):
    """Extract one JPEG frame at timestamp t."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", video_path,
         "-frames:v", "1", "-q:v", str(quality), "-y", out_path],
        capture_output=True, timeout=120)
    return out_path


def extract_frames(video_path, out_dir=None, interval=DEFAULT_INTERVAL):
    """Extract all sampled frames; returns (duration, times, frame_paths)."""
    duration = get_duration(video_path)
    times = frame_times(duration, interval)
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="mugc_frames_")
    paths = []
    for t in times:
        p = os.path.join(out_dir, "frame_%03d.jpg" % t)
        paths.append(extract_frame(video_path, t, p))
    return duration, times, paths


def main():
    ap = argparse.ArgumentParser(description="Sample frames the way the MUGC VG pipeline does.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    duration, times, paths = extract_frames(args.video, args.out_dir, args.interval)
    print("duration      : %.3f s" % duration)
    print("floor(dur)    : %d" % int(duration))
    print("frame times   : %s" % times)
    print("frames written: %d" % len(paths))
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
