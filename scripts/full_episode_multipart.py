"""Bounded streaming multipart upload, shared with the subtask collector."""
import base64
import hashlib
import xml.etree.ElementTree as ET

class MultipartArchive:
    def __init__(self, urls, request, part_bytes=128*1024**2):
        self.urls,self.request,self.part_bytes=urls,request,part_bytes
        self.buffer=bytearray();self.parts=[];self.hash=hashlib.sha256();self.count=0
    def writable(self):return True
    def write(self, data):
        self.hash.update(data);self.count+=len(data);self.buffer.extend(data)
        while len(self.buffer)>=self.part_bytes:
            self.send(bytes(self.buffer[:self.part_bytes]));del self.buffer[:self.part_bytes]
        return len(data)
    def send(self, body):
        number=len(self.parts)+1
        if number>len(self.urls['parts']):raise ValueError('Archive exceeds its frozen object budget')
        md5=hashlib.md5(body).digest()
        _,headers=self.request(self.urls['parts'][number-1],method='PUT',body=body,
            headers={'Content-Length':str(len(body)),'Content-MD5':base64.b64encode(md5).decode()})
        headers={k.lower():v for k,v in headers.items()}
        etag=headers.get('etag')
        if not etag:raise ValueError('Stored archive part lacks its receipt')
        self.parts.append((number,etag,md5))
    def finish(self):
        if self.buffer:self.send(bytes(self.buffer));self.buffer.clear()
        xml=ET.Element('CompleteMultipartUpload')
        for number,etag,_ in self.parts:
            part=ET.SubElement(xml,'Part');ET.SubElement(part,'PartNumber').text=str(number);ET.SubElement(part,'ETag').text=etag
        body=ET.tostring(xml)
        response,_=self.request(self.urls['complete'],method='POST',body=body,headers={'Content-Type':'application/xml'})
        tree=ET.fromstring(response)
        if tree.tag.rsplit('}',1)[-1]=='Error':raise ValueError('Object store rejected multipart completion')
        _,head=self.request(self.urls['head'],method='HEAD')
        head={k.lower():v for k,v in head.items()}
        if int(head.get('content-length','-1'))!=self.count:raise ValueError('Archived byte count differs')
        expected=hashlib.md5(b''.join(p[2] for p in self.parts)).hexdigest()+'-'+str(len(self.parts))
        actual=head.get('etag','').strip('"').lower()
        if actual!=expected:raise ValueError('Object-store multipart digest differs')
        return dict(object_key=self.urls['key'],bytes=self.count,sha256=self.hash.hexdigest(),
            etag=head.get('etag'),parts=len(self.parts),part_content_md5_verified=True)
