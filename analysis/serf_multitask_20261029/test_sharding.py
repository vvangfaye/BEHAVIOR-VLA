import unittest
from sharding import eight_shards


class ShardingTest(unittest.TestCase):
    def test_eight_workers_preserve_all_eighty_cases_once(self):
        tasks=[{'task_id':21,'instance_ids':list(range(100,120))},{'task_id':26,'instance_ids':list(range(200,220))}]
        shards=eight_shards(tasks);self.assertEqual(len(shards),8)
        cases=[]
        for shard in shards:
            task=next(t for t in tasks if t['task_id']==shard['task_id'])
            self.assertEqual(len(shard['indices']),5)
            for i in shard['indices']:
                for condition in ['image','serf']:cases.append((task['task_id'],task['instance_ids'][i],condition))
        expected={(t['task_id'],instance,c) for t in tasks for instance in t['instance_ids'] for c in ['image','serf']}
        self.assertEqual(len(cases),80);self.assertEqual(set(cases),expected);self.assertEqual(len(cases),len(set(cases)))


if __name__=='__main__':unittest.main()
