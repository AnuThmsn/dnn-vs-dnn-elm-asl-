import os
import sys
import psutil

curr_pid = os.getpid()
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
    if p.info['name'] == 'python.exe' and p.info['pid'] != curr_pid:
        mem = p.info['memory_info'].rss / (1024**2)
        cmd = ' '.join(p.info['cmdline'] or [])
        print(f"PID {p.info['pid']}: {mem:.1f} MB | {cmd[:100]}")

