#!/bin/bash
# encode_frames.sh FRAMES_DIR NAME — build real-time MP4s from record_frames.py output (2026-10-17).
# Each frame is shown until the next one's timestamp (ffmpeg concat with durations), so the video
# runs at wall-clock speed whatever the capture rate. Writes NAME_rgb.mp4, NAME_annotated.mp4 and
# NAME_side_by_side.mp4 (RGB | annotated, aligned by stamp) into FRAMES_DIR. Meant for a machine
# with spare cores (dual-a4500), not the robot PC.
set -e
D=$(realpath "$1"); NAME=$2
for k in rgb annotated; do
  python3 - "$D/$k" > "$D/$k.concat" <<'EOF'
import os, sys
d = sys.argv[1]
fs = sorted(os.listdir(d), key=lambda f: int(f.split('.')[0]))
ts = [int(f.split('.')[0]) for f in fs]
for i, f in enumerate(fs):
    dur = (ts[i + 1] - ts[i]) / 1e9 if i + 1 < len(fs) else 0.1
    print(f"file '{os.path.join(d, f)}'\nduration {max(dur, 0.001):.6f}")
print(f"file '{os.path.join(d, fs[-1])}'")
EOF
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$D/$k.concat" -vf "fps=30,format=yuv420p" \
    -c:v libx264 -preset medium -crf 22 -movflags +faststart "$D/${NAME}_$k.mp4"
done
# Align: start both at the later of the two first stamps
R0=$(ls "$D/rgb" | sort -n | head -1 | cut -d. -f1); A0=$(ls "$D/annotated" | sort -n | head -1 | cut -d. -f1)
OFF=$(python3 -c "print(($A0-$R0)/1e9)")
python3 -c "import sys; sys.exit(0 if $OFF >= 0 else 1)" && SR="-ss $OFF" SA="" || { SR=""; SA="-ss $(python3 -c "print(-($OFF))")"; }
ffmpeg -y -loglevel error $SR -i "$D/${NAME}_rgb.mp4" $SA -i "$D/${NAME}_annotated.mp4" -filter_complex \
  "[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]hstack=inputs=2:shortest=1" \
  -c:v libx264 -preset medium -crf 22 -pix_fmt yuv420p -movflags +faststart "$D/${NAME}_side_by_side.mp4"
ls -la "$D"/${NAME}_*.mp4
