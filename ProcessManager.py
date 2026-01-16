import multiprocessing
import signal
import sys
import time
from VolumeWorker import VolumeWorker
from SpatialScheduler import SpatialScheduler
import asyncio
import traceback


class ProcessManager:
    def __init__(self, worker_num=4, port=5555):
        self.worker_num = worker_num
        self.bind_addr = f"tcp://127.0.0.1:{port}"
        self.procs = []

    def start_scheduler(self):
        """Scheduler 是异步的，单独一个进程跑 asyncio loop"""
        scheduler = SpatialScheduler(self.bind_addr)
        asyncio.run(scheduler.run())

    def start_worker(self, idx):
        """Worker 是同步循环"""
        worker = VolumeWorker(self.bind_addr, idx)
        worker.run()

    def start_cluster(self):
        print("=== Starting CloudVolume Cluster ===")
        
        # 1. 启动 Scheduler
        p_sched = multiprocessing.Process(target=self.start_scheduler, name="Scheduler")
        p_sched.start()
        self.procs.append(p_sched)
        time.sleep(1) # 等待 Scheduler bind 端口
        
        # 2. 启动 Workers
        for i in range(self.worker_num):
            p_worker = multiprocessing.Process(
                target=self.start_worker, 
                args=(i,), 
                name=f"Worker-{i}"
            )
            p_worker.start()
            self.procs.append(p_worker)
        
        print(f"=== Cluster Ready: 1 Scheduler + {self.worker_num} Workers ===")

    def graceful_shutdown(self, signum, frame):
        print("\n=== Shutting down cluster... ===")
        for p in self.procs:
            if p.is_alive():
                p.terminate()
                p.join()
        print("=== All processes stopped. Bye. ===")
        sys.exit(0)

    def monitor(self):
        """主线程阻塞监控"""
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        
        while True:
            # 简单的健康检查
            dead_procs = [p for p in self.procs if not p.is_alive()]
            if dead_procs:
                print(f"Warning: {len(dead_procs)} processes died!")
                # 在这里可以添加重启逻辑 restart_worker()
            time.sleep(5)

# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    # 请替换为真实的 CloudVolume 路径
    try:
        manager = ProcessManager(worker_num=4, port=5555)
        manager.start_cluster()
        manager.monitor()
    except Exception as e:
        traceback.print_exc()
