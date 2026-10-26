import io
import json
import tarfile
import unittest
from recovery_archives import completed_archives


class ClientError(Exception):
    def __init__(self,code):
        self.response={'Error':{'Code':code}}


class Client:
    class exceptions:
        ClientError=ClientError

    def __init__(self,objects):self.objects=objects

    def head_object(self,Bucket,Key):
        if Key not in self.objects:raise ClientError('NoSuchKey')

    def get_object(self,Bucket,Key):return {'Body':io.BytesIO(self.objects[Key])}


def archive(instance,condition,folder_instance=None):
    stream=io.BytesIO()
    payload=json.dumps({'complete':True,'instance':instance,'condition':condition}).encode()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        item=tarfile.TarInfo(f'{instance if folder_instance is None else folder_instance}_{condition}/result.json');item.size=len(payload)
        tar.addfile(item,io.BytesIO(payload))
    return stream.getvalue()


class RecoveryTest(unittest.TestCase):
    def test_prior_baseline_survives_multiple_recoveries(self):
        client=Client({'g2/episodes/282_image.tar.gz':archive(282,'image'),
                       'g3/episodes/282_serf.tar.gz':archive(282,'serf')})
        completed,items=completed_archives(client,'bucket',
            [{'result_key':f'g{i}/status.json'} for i in [3,2,1]],
            [(282,'image'),(282,'serf'),(281,'image')],lambda key:key)
        self.assertEqual(set(completed),{'282_image','282_serf'})
        self.assertEqual([x['key'] for x in items],['g2/episodes/282_image.tar.gz','g3/episodes/282_serf.tar.gz'])

    def test_mismatched_instance_is_rejected(self):
        client=Client({'g3/episodes/282_image.tar.gz':archive(281,'image',folder_instance=282)})
        with self.assertRaises(AssertionError):
            completed_archives(client,'bucket',[{'result_key':'g3/status.json'}],[(282,'image')],str)

    def test_newest_completed_archive_wins(self):
        client=Client({f'g{i}/episodes/282_image.tar.gz':archive(282,'image') for i in [2,3]})
        _,items=completed_archives(client,'bucket',[{'result_key':f'g{i}/status.json'} for i in [3,2]],[(282,'image')],str)
        self.assertEqual(items[0]['key'],'g3/episodes/282_image.tar.gz')


if __name__=='__main__':unittest.main()
