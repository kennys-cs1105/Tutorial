from multiprocessing import Process
import os
import time

def run():
    while True:
        print(f"current process id : {os.getpid()}")
        print(f"parent process id of current process {os.getpid()} : {os.getppid()}")
        time.sleep(2)

if __name__ == "__main__":
    p = Process(target=run)
    p.start()