import subprocess,io,json
from pathlib import Path
from PIL import Image,ImageDraw
root=Path('analysis/behavior_handoff_visual_review_20261020/living_room')
base=Path('analysis/retraining_20261016_quality_taskbalanced_v1/confirmation_selected_vs_baseline/video_review/media')
ff='/tmp/b1k-video-review-libs/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
def frame(run,t):
 p=base/f'{run}-18/tidying_living_room_65_0.mp4'
 r=subprocess.run([ff,'-hide_banner','-loglevel','error','-ss',str(t),'-i',str(p),'-frames:v','1','-f','image2pipe','-vcodec','png','-threads','1','-'],capture_output=True,check=True)
 return Image.open(io.BytesIO(r.stdout)).convert('RGB')
def sheet(name,times):
 W,H=672,448
 im=Image.new('RGB',(W*2,(H+25)*len(times)),'white');d=ImageDraw.Draw(im)
 for row,t in enumerate(times):
  for col,run in enumerate(['baseline','candidate']):
   img=frame(run,t);img.save(root/f'{run}_{t:06.1f}.jpg',quality=95)
   im.paste(img,(col*W,row*(H+25)+25));d.text((col*W+6,row*(H+25)+6),f'{run} | {t:.1f} sec',fill='black')
 im.save(root/f'{name}.jpg',quality=94)
 print(root/f'{name}.jpg')
if __name__=='__main__':
 import sys
 sheet(sys.argv[1],list(map(float,sys.argv[2].split(','))))
