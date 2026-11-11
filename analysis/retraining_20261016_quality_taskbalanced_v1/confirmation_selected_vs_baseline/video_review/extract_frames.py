"""Make timestamped paired contact sheets from verified completed videos."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image, ImageDraw

sys.path.insert(0, '/tmp/b1k-video-review-libs')
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def frame(path, seconds, width):
    result = subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-ss', str(seconds),
                             '-i', str(path), '-frames:v', '1', '-vf', f'scale={width}:-1',
                             '-f', 'image2pipe', '-vcodec', 'png', '-threads', '1', '-'],
                            capture_output=True, timeout=45, check=True)
    return Image.open(io.BytesIO(result.stdout)).convert('RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--times', default='0,30,90,180,270,360,450,530')
    parser.add_argument('--width', type=int, default=760)
    args = parser.parse_args()
    times = list(map(float, args.times.split(',')))
    output = ROOT / 'frames'
    output.mkdir(exist_ok=True)
    images = []
    for second in times:
        pair = []
        for role in ('baseline', 'candidate'):
            path = next((ROOT / 'media' / f'{role}-{args.index:02d}').glob('*.mp4'))
            report = json.loads((ROOT.parent / 'reports' / f'{role}-{args.index:02d}.json').read_text())
            duration = report['record']['steps'] / 30
            actual = min(second, max(0, duration - .1))
            img = frame(path, actual, args.width)
            name = f'{role}-{args.index:02d}-t{actual:.1f}.jpg'
            img.save(output / name, quality=92)
            caption = f"{role} | t={actual:.1f}s" + (' [ENDED]' if second >= duration else '')
            pair.append((img, caption))
        images.append(pair)
    rowheight = max(im.height for pair in images for im, _ in pair) + 25
    sheet = Image.new('RGB', (args.width * 2, rowheight * len(images)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, pair in enumerate(images):
        for col, (img, caption) in enumerate(pair):
            draw.text((col * args.width + 5, row * rowheight + 4), caption, fill='black')
            sheet.paste(img, (col * args.width, row * rowheight + 25))
    target = output / f'pair-{args.index:02d}-contact.jpg'
    sheet.save(target, quality=92)
    print(json.dumps({'contact_sheet': str(target), 'times': times, 'dimensions': sheet.size}))


if __name__ == '__main__':
    main()
