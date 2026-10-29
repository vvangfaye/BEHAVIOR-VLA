"""Keep each baseline/SERF pair together while using eight independent workers."""
def eight_shards(tasks):
    assert len(tasks)==2 and all(len(t['instance_ids'])==20 for t in tasks)
    return [dict(shard=block*2+offset,task_id=task['task_id'],indices=list(range(block*5,(block+1)*5)))
            for block in range(4) for offset,task in enumerate(tasks)]
