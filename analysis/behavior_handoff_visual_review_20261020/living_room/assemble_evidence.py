from PIL import Image,ImageDraw
from pathlib import Path
p=Path(__file__).parent
items=[('baseline',120,'Newspaper carried away from initial floor location'),('candidate',320,'Newspaper held; plant still at the wall/floor'),('baseline',400,'Plant lifted and transported in right hand'),('baseline',468,'Plant near table edge; release not yet established'),('baseline',485,'Plant below table level / floor beside table'),('candidate',628,'Still at wall with newspaper; no plant transport')]
w,h=672,448
out=Image.new('RGB',(2*w,3*(h+48)),'white');d=ImageDraw.Draw(out)
for n,(run,t,caption) in enumerate(items):
 x=(n%2)*w;y=(n//2)*(h+48)
 d.text((x+6,y+5),f'{run} | t={t:.1f} s',fill='black')
 d.text((x+6,y+25),caption,fill='black')
 out.paste(Image.open(p/f'{run}_{t:06.1f}.jpg'),(x,y+48))
out.save(p/'evidence_6frames.jpg',quality=95)
