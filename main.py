from ClientProxy import ClientProxy, AutoReleaseArray
import traceback
import time, random
LOCAL_CLONE = '/CX/neuro_tracking/xinr/cloudvolume_test/Cloudvolume_replay'
import os
import sys
if os.path.exists(LOCAL_CLONE):
    sys.path.insert(0, LOCAL_CLONE)

from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox 

def main():
    # 假设此时有一个 Scheduler 在监听 5555 端口
    cv = CloudVolume(
            '/CX/neuro_tracking/fafb-ffn1', 
            mip=0,
            fill_missing=True,
            cache=True,
            log_path="/dev/shm/222.log",            # 开启磁盘缓存
            lru_bytes= 80* 1024**2, 
            partial_decompress_parallel=20
        )
    client = ClientProxy("tcp://127.0.0.1:5555", cv)
    center = [26676, 8024, 3811]
    size = [1000, 1000, 10]
    base = [0, 0, 0]
    count = 10
    
    # 模拟切片调用
    try:
        while count:
            # base = [random.randint(-1000,1000), random.randint(-1000,1000), random.randint(-1000,1000)]
            count = count - 1
            # 这一行会：申请内存 -> 发请求 -> 阻塞等待 -> 返回自动回收数组
            time_start = time.perf_counter()
            vol_data = client[center[0] + base[0] : center[0] + base[0] + size[0], center[1] + base[1]: center[1] + base[1] + size[1], center[2] + base[2]: center[2] + base[2] + size[2]]
            
            print(f"time_cost = {time.perf_counter() - time_start}")
            print(f"Got data shape: {vol_data.shape}")
            print(f"Data at [0,0,0]: {vol_data[0,0,0]}")

    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()