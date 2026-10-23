import base64,json,pathlib,traceback
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/checkpoints/pi05_b1k_full100_joint_1688977_aoss_20261010')
result={'source':str(root),'exists':root.exists(),'entries':[p.name for p in root.iterdir()] if root.exists() else []}
try:(root/'params/_METADATA').read_bytes()
except Exception as e:result['exact_old_bootstrap_check_error']={'type':type(e).__name__,'message':str(e)}
marker=root/'AFS_ARCHIVED.json'
if marker.exists():result['archive_pointer']=json.loads(marker.read_text())
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KPROBE_END',flush=True)
