"""Local side-by-side four-times-speed excerpts, original episodes preserved."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, '/tmp/b1k-video-review-libs')
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
WINDOWS = {19: [(100, 140), (200, 250), (320, 350)],
           11: [(100, 140), (210, 250), (320, 360), (410, 440)],
           18: [(90, 120), (270, 310), (395, 450), (590, 620)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    target = ROOT / 'highlights'
    target.mkdir(exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    videos = [next((ROOT / 'media' / f'{r}-{args.index:02d}').glob('*.mp4'))
              for r in ('baseline', 'candidate')]
    parts = []
    for number, (start, end) in enumerate(WINDOWS[args.index]):
        output = target / f'pair-{args.index:02d}-part-{number}.mp4'
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y']
        for video in videos:
            command += ['-ss', str(start), '-t', str(end - start), '-i', str(video)]
        filters = ('[0:v]scale=640:-2,setpts=(PTS-STARTPTS)/4[a];'
                   '[1:v]scale=640:-2,setpts=(PTS-STARTPTS)/4[b];[a][b]hstack=inputs=2,fps=30[out]')
        command += ['-filter_complex', filters, '-map', '[out]', '-an', '-c:v', 'libx264',
                    '-preset', 'fast', '-crf', '24', '-threads', '2', '-movflags', '+faststart', str(output)]
        subprocess.run(command, check=True, capture_output=True, timeout=180)
        parts.append(output)
    listing = target / f'pair-{args.index:02d}-concat.txt'
    listing.write_text(''.join("file '" + p.name + "'\n" for p in parts))
    final = target / f'pair-{args.index:02d}-highlight.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
                    '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(final)],
                   capture_output=True, check=True, timeout=30)
    (target / f'pair-{args.index:02d}-segments.json').write_text(json.dumps({
        'left': 'baseline', 'right': 'expert_mse', 'playback_speed': 4,
        'simulation_time_windows_seconds': WINDOWS[args.index], 'full_episodes_preserved': True}, indent=2) + '\n')
    print(json.dumps({'highlight': str(final), 'bytes': final.stat().st_size}))


if __name__ == '__main__':
    main()
