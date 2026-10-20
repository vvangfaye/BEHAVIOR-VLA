"""One explicitly recorded retry after the first CPU job failed before Python output."""
import collect

collect.NAME='b1k-storehoney-audit-1020-r1'
collect.HERE=collect.HERE/'attempt02'
collect.HERE.mkdir(exist_ok=True)

if __name__=='__main__':
    collect.main()
