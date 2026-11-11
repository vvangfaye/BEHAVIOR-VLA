import aoss_io, json, subprocess, base64, datetime
probe = open('probe.sh').read()
CLI = '/home/faye/.local/bin/sco'
for spec, T, W, Q in [("h100", "b1k-update-composition-1022-plumbing06h100", "n6ls.iu.i40.1", "reserved"),
                      ("5090", "b1k-repeat1688-1019-batch-00", "n12lp.nn.i10a.1", "spot")]:
    key = f"cluster_probe_20261104/{spec}/probe.txt"
    url = aoss_io.presign("put_object", key)
    b64 = base64.b64encode(probe.encode()).decode()
    cmd = (f"echo {b64} | base64 -d > /tmp/probe.sh && timeout 300 bash /tmp/probe.sh > /tmp/probe_out.txt 2>&1; "
           f"curl -s -m 60 -X PUT --upload-file /tmp/probe_out.txt '{url}'; echo upload_rc=$?")
    name = f"b1k-probe2-{spec}-" + datetime.datetime.now().strftime("%m%d%H%M")
    r = subprocess.run([CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space', '--copy-job-name=' + T, '--name=' + name,
                        '--job-name=' + name, '--worker-spec=' + W, '--worker-nodes=1', '--quota-type=' + Q, '--retry-times=0',
                        '--enable-fault-tolerance=false', '--enable-anomaly-detection=false', '--command=' + cmd],
                       capture_output=True, text=True, timeout=90)
    json.dump({"name": name, "template": T, "spec": W, "quota": Q, "aoss_key": key, "rc": r.returncode,
               "submitted": datetime.datetime.now().isoformat()}, open(f"receipt2_{spec}.json", "w"), indent=1)
    tail = (r.stdout + r.stderr).strip().splitlines()
    print(spec, name, "rc", r.returncode, tail[-1][:120] if tail else "")
