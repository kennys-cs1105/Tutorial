from time import sleep, time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

"""
比较多线程和单线程的性能差异
以测试图片下载为例
"""

def download_image(url):
    sleep(1)
    print(f"{url} download complete...")


if __name__ == "__main__":

    # """单线程"""
    # start_time = time()
    # for i in range(5):
    #     download_image(f"url{i}")
    # end_time = time()
    # print(f"单线程耗时: {end_time - start_time}")

    """多线程"""
    start_time = time()
    threads = []
    
    for i in range(5):
        t = Thread(target=download_image, args=[f"url_{i}"])
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    end_time = time()

    print(f"多线程耗时: {end_time - start_time}")

