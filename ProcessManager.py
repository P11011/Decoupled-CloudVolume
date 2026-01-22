import multiprocessing
import signal
import sys
import time
import asyncio
import traceback
from VolumeWorker import VolumeWorker
from SpatialScheduler import SpatialScheduler

class ProcessManager:
    def __init__(self, port=5555):
        self.bind_addr = f"tcp://127.0.0.1:{port}"
        self.procs = []

    def start_scheduler(self):
        scheduler = SpatialScheduler(self.bind_addr)
        asyncio.run(scheduler.run())

    def start_worker(self, idx, parallel_config):
        worker = VolumeWorker(self.bind_addr, idx, parallel=parallel_config)
        worker.run()

    def start_cluster(self, config_list):
        print("=== Starting CloudVolume Cluster ===")
        print(f"Config Plan: {config_list}")
        
        p_sched = multiprocessing.Process(target=self.start_scheduler, name="Scheduler")
        p_sched.start()
        self.procs.append(p_sched)
        time.sleep(1) # 等待 Scheduler bind 端口
        
        worker_idx = 0  # 全局 Worker 编号计数器

        for p_level, count in config_list:
            print(f"-> Spawning {count} workers with parallel={p_level}...")
            
            for _ in range(count):
                p_worker = multiprocessing.Process(
                    target=self.start_worker, 
                    args=(worker_idx, p_level), 
                    name=f"Worker-{worker_idx}-P{p_level}"
                )
                p_worker.start()
                self.procs.append(p_worker)
                worker_idx += 1
        
        print(f"=== Cluster Ready: 1 Scheduler + {worker_idx} Workers ===")

    def graceful_shutdown(self, signum, frame):
        print("\n=== Shutting down cluster... ===")
        for p in self.procs:
            if p.is_alive():
                p.terminate()
                p.join()
        sys.exit(0)

    def monitor(self):
        """主线程阻塞监控"""
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        
        while True:
            dead_procs = [p for p in self.procs if not p.is_alive()]
            if dead_procs:
                print(f"Warning: {len(dead_procs)} processes died!")
            time.sleep(5)

# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    try:
        manager = ProcessManager(port=5555)
        
        # 配置列表: [ [parallel参数, 进程数量], ... ]
        # 示例含义:
        # - 启动 4 个 parallel=4 的进程 (适合处理小块请求，轻量级)
        # - 启动 2 个 parallel=20 的进程 (适合处理大块请求，利用多核解压)
        my_config = [
            [20, 60],   # A=4, B=4
        ]
        
        manager.start_cluster(my_config)
        manager.monitor()
    except Exception as e:
        traceback.print_exc()