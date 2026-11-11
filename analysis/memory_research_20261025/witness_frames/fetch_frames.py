"""Read selected frames from existing public videos, with bounded HTTP ranges."""
import io
import json
from pathlib import Path

import av
import requests


class RangeReader(io.RawIOBase):
    def __init__(self, url, budget=64 * 1024**2, block=1024**2):
        self.url, self.budget, self.block = url, budget, block
        self.session = requests.Session()
        response = self.session.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        self.length = int(response.headers['Content-Length'])
        self.pos, self.used, self.cache = 0, 0, {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [offset, self.pos + offset, self.length + offset][whence]
        if self.pos < 0:
            raise ValueError('Negative seek')
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.pos
        size = min(size, self.length - self.pos)
        output = bytearray()
        while size > 0:
            number, within = divmod(self.pos, self.block)
            if number not in self.cache:
                start = number * self.block
                end = min(start + self.block, self.length) - 1
                if self.used + end - start + 1 > self.budget:
                    raise RuntimeError('Range-read budget exhausted')
                response = self.session.get(
                    self.url, headers={'Range': f'bytes={start}-{end}'},
                    stream=True, timeout=60,
                )
                response.raise_for_status()
                if response.status_code != 206:
                    response.close()
                    raise RuntimeError('Server did not honor bounded range request')
                if response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.length}':
                    response.close()
                    raise RuntimeError('Unexpected content range')
                data = response.raw.read(end - start + 1)
                response.close()
                if len(data) != end - start + 1:
                    raise RuntimeError('Incomplete range')
                self.used += len(data)
                self.cache[number] = data
            data = self.cache[number]
            take = min(size, len(data) - within)
            output.extend(data[within:within + take])
            self.pos += take
            size -= take
        return bytes(output)


def main():
    out = Path(__file__).resolve().parent
    sources = {
        'head': ('observation.rgb.zed_link_camera_0', 14, 1686.3666666666666),
        'right': ('observation.rgb.right_realsense_link_camera_0', 5, 4944.9),
    }
    results = []
    for camera, (key, file_index, start_pts) in sources.items():
        relative = f'videos/{key}/chunk-053/file-{file_index:03d}.mp4'
        url = 'https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/resolve/main/' + relative
        reader = RangeReader(url)
        container = av.open(reader)
        stream = container.streams.video[0]
        entries = []
        for frame_index in (3900, 4101, 4395, 4407, 4410, 4440):
            target = start_pts + frame_index / 30
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                actual = float(frame.pts * stream.time_base)
                if actual + 1e-5 >= target:
                    filename = f'{camera}_{frame_index}.png'
                    frame.to_image().save(out / filename)
                    entries.append({'episode_frame': frame_index, 'requested_pts': target,
                                    'decoded_pts': actual, 'file': filename})
                    break
            else:
                raise RuntimeError(f'No frame at {camera}:{frame_index}')
        container.close()
        result = {'camera': camera, 'relative_path': relative,
                  'video_bytes': reader.length, 'downloaded_bytes': reader.used,
                  'frames': entries}
        results.append(result)
        print(json.dumps({'camera': camera, 'downloaded_bytes': reader.used, 'frames': len(entries)}), flush=True)
    (out / 'receipt.json').write_text(json.dumps({'episode_index': 10710, 'fps': 30,
                                                'sources': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
