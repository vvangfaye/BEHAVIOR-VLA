import aoss_io, json, subprocess, base64, datetime
CLI = '/home/faye/.local/bin/sco'
key = "cluster_probe_20261104/net/probe_net.txt"
url = aoss_io.presign("put_object", key)
b64 = base64.b64encode(open('probe_net.sh').read().encode()).decode()
cmd = (f"echo {b64} | base64 -d > /tmp/p.sh && timeout 420 bash /tmp/p.sh > /tmp/p_out.txt 2>&1; "
       f"curl -s -m 60 -X PUT --upload-file /tmp/p_out.txt '{url}'")
name = "b1k-probe-net-5090-" + datetime.datetime.now().strftime("%m%d%H%M")
r = subprocess.run([CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space', '--copy-job-name=b1k-repeat1688-1019-batch-00',
                    '--name=' + name, '--job-name=' + name, '--worker-spec=n12lp.nn.i10a.1', '--worker-nodes=1', '--quota-type=spot',
                    '--retry-times=0', '--enable-fault-tolerance=false', '--enable-anomaly-detection=false', '--command=' + cmd],
                   capture_output=True, text=True, timeout=90)
json.dump({"name": name, "aoss_key": key, "rc": r.returncode, "submitted": datetime.datetime.now().isoformat()}, open("receipt_net.json", "w"))
print(name, r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1][:100])
