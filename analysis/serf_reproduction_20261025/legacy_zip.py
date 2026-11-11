"""HTTP-range ZIP inspection and CRC-verified extraction of selected assets."""
import io,json,struct,urllib.request,zipfile,zlib


class RangeFile(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        data=fetch_range(self.url,self.pos,self.pos+n-1);self.pos+=len(data);return data


def fetch_range(url,start,end):
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=90) as r:
        assert r.status==206,(r.status,start,end)
        assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),r.headers['Content-Range']
        data=r.read(end-start+2)
    assert len(data)==end-start+1,(len(data),end-start+1)
    return data


def index(url,size):
    with zipfile.ZipFile(RangeFile(url,size)) as z:
        return [{'name':f.filename,'size':f.file_size,'compressed_size':f.compress_size,'crc':f.CRC,
                 'offset':f.header_offset,'compression':f.compress_type,'directory':f.is_dir()} for f in z.infolist()]


def decode_entry(info,chunk,start):
    at=info['offset']-start
    header=struct.unpack('<IHHHHHIIIHH',chunk[at:at+30]);assert header[0]==0x04034b50
    begin=at+30+header[-2]+header[-1]
    data=chunk[begin:begin+info['compressed_size']]
    if info['compression']==zipfile.ZIP_DEFLATED:data=zlib.decompress(data,-15)
    elif info['compression']!=zipfile.ZIP_STORED:raise ValueError('Unsupported ZIP compression')
    assert len(data)==info['size'] and (zlib.crc32(data)&0xffffffff)==info['crc'],info['name']
    return data


if __name__=='__main__':
    from pathlib import Path
    HERE=Path(__file__).resolve().parent
    extra=json.loads((HERE/'extra_asset_manifest.json').read_text())['zip_info']
    name='behavior-1k-assets-3.7.2rc1.zip';record=next(x for x in extra['siblings'] if x['rfilename']==name)
    url=f"https://huggingface.co/datasets/behavior-1k/zipped-datasets/resolve/{extra['sha']}/{name}"
    records=index(url,record['size'])
    result={'url':url,'size':record['size'],'archive_sha256':record['lfs']['sha256'],'entries':records}
    (HERE/'legacy_asset_zip_index.json').write_text(json.dumps(result,indent=2)+'\n')
    print('entries',len(records),'first',[x['name'] for x in records[:15]])
    totals={}
    for x in records:
        top=x['name'].split('/')[0];totals[top]=totals.get(top,0)+x['size']
    print(totals)
