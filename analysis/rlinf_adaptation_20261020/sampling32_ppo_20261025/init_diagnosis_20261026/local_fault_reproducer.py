import os,faulthandler,threading,time,resource,sys
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
stop=threading.Event()
def rec(x):
 a=1
 b=2
 c=3
 if x>0:return rec(x-1)
def churn():
 while not stop.is_set():rec(123)
t=threading.Thread(target=churn);t.start()
with open(os.devnull,'w') as f:
 if sys.argv[1]=='enabled':faulthandler.dump_traceback_later(.0001,repeat=True,file=f)
 try:time.sleep(3)
 finally:
  faulthandler.cancel_dump_traceback_later();stop.set();t.join()
print(sys.version)
