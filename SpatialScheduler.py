import utils
import zmq
import zmq.asyncio
import msgpack
import time
import asyncio
from collections import deque, defaultdict

class SpatialScheduler:
    def __init__(self, bind_addr, history_len=5):
        self.bind_addr = bind_addr
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        
        # Worker 管理
        self.workers = set()  # 存活的 Worker ID 集合
        self.worker_list = []
        self.rr_counter = 0
        self.worker_history = defaultdict(lambda: deque(maxlen=history_len)) 
        self.worker_load = defaultdict(int) # 记录每个 Worker 当前积压的任务数
        
        # [新增] 进程亲和性映射表: PID (str) -> WorkerID (bytes)
        self.process_map = {} 
        
        self.ready_event = asyncio.Event()

    async def run(self):
        self.socket.bind(self.bind_addr)
        print(f"[Scheduler] Listening on {self.bind_addr}")
        
        while True:
            # 接收多帧消息: [Identity, Empty, Payload]
            frames = await self.socket.recv_multipart()
            identity = frames[0]
            payload = msgpack.unpackb(frames[-1])
            
            msg_type = payload.get('type')
            
            if msg_type == 'READY':
                self._handle_worker_ready(identity)
            elif msg_type == 'READ':
                await self._dispatch_request(identity, payload, frames)
            elif msg_type == 'RESULT':
                # 注意：Result 返回时 identity 是 WorkerID
                await self._forward_result(identity, payload)
            else:
                print(f"[Scheduler] Unknown message type: {msg_type}")

    def _handle_worker_ready(self, worker_id):
        if worker_id not in self.workers:
            self.workers.add(worker_id)
            self.worker_load[worker_id] = 0
            self.worker_list = sorted(list(self.workers))
            print(f"[Scheduler] Worker registered: {worker_id}")
            self.ready_event.set()

    async def old_dispatch_request(self, client_id, payload, raw_frames):
        """
        策略：Process Affinity + Least Load Fallback
        优先将同一进程的请求发给同一 Worker；如果该 Worker 忙碌，则重新分配给负载最小的 Worker。
        """
        time1 = time.time()
        if not self.workers:
            print("[Scheduler] No workers available! Waiting...")
            await self.ready_event.wait()

        # 1. 从 req_id 解析 PID
        # req_id format: "{pid}_req_{uuid}"
        req_id = payload.get('req_id', '')
        try:
            pid = req_id.split('_')[0]
        except IndexError:
            # 如果格式不对，用 client_id 做 fallback
            pid = client_id

        # 2. 获取当前集群负载状态
        min_load = min(self.worker_load[w] for w in self.workers)
        
        # [配置] 忙碌阈值：如果目标 Worker 的负载比最闲的 Worker 高出多少，视为"不空闲/忙碌"
        # 设为 0 表示严格追求绝对空闲；设为 2 表示允许少量排队以换取缓存
        LOAD_TOLERANCE = 2 

        # 3. 决策最佳 Worker
        best_worker = None
        target_worker = self.process_map.get(pid)

        # 逻辑：
        # A. 如果有绑定的 Worker，且该 Worker 存活，且该 Worker 相对空闲 -> 保持绑定
        # B. 否则 (新进程 或 原Worker太忙) -> 选负载最小的 -> 更新绑定
        
        if target_worker and target_worker in self.workers:
            target_load = self.worker_load[target_worker]
            # 判定是否"空闲" (这里定义为: 负载不超过最小负载 + 容忍度)
            if target_load <= min_load + LOAD_TOLERANCE:
                best_worker = target_worker
            else:
                # 原 Worker 太忙，切换到最闲的 Worker
                best_worker = min(self.workers, key=lambda w: self.worker_load[w])
                self.process_map[pid] = best_worker # 更新绑定
        else:
            # 新进程 或 原 Worker 已下线
            best_worker = min(self.workers, key=lambda w: self.worker_load[w])
            self.process_map[pid] = best_worker # 建立绑定

        # 4. 更新状态
        bbox = payload['bbox']
        self.worker_history[best_worker].append(bbox)
        self.worker_load[best_worker] += 1 
        
        # 转发
        payload['client_id'] = client_id
        
        # 调试日志
        # status = "Affinity" if best_worker == target_worker else "Rebalance"
        # print(f"[Sched] {pid} -> {best_worker} ({status}, Load: {self.worker_load[best_worker]})")
        
        new_msg = [best_worker, b"", msgpack.packb(payload)]
        await self.socket.send_multipart(new_msg)

    async def _forward_result(self, worker_id, payload):
        """
        收到 Worker 结果，转发回 Client，并减少 Worker 负载
        """
        if worker_id in self.worker_load:
            self.worker_load[worker_id] = max(0, self.worker_load[worker_id] - 1)
            
        client_id = payload['client_id']
        await self.socket.send_multipart([client_id, b"", msgpack.packb(payload)])


    def _get_round_robin_worker(self):
        """
        O(1) 复杂度的轮询获取。
        直接读取缓存好的 worker_list。
        """
        # 注意：外部的 dispatch 已经 check 了 if not self.workers，所以这里列表肯定不为空
        
        # 1. 取模获取索引
        idx = self.rr_counter % len(self.worker_list)
        
        # 2. 获取 Worker ID
        worker_id = self.worker_list[idx]
        
        # 3. 计数器自增
        self.rr_counter += 1
        
        return worker_id

    # [修改方法] 使用轮询的分发逻辑
    async def _dispatch_request(self, client_id, payload, raw_frames):
        # 1. 等待 Worker 可用
        if not self.workers:
            print("[Scheduler] No workers available! Waiting...")
            await self.ready_event.wait()

        # 2. [修改点] 调用轮询方法获取 Worker
        best_worker = self._get_round_robin_worker()

        # 3. 更新状态 (保持原有逻辑用于统计)
        bbox = payload.get('bbox')
        if bbox:
            self.worker_history[best_worker].append(bbox)
        
        self.worker_load[best_worker] += 1 
        
        # 4. 转发消息
        payload['client_id'] = client_id
        
        # 调试日志：可以看到分配是严格轮询的
        # print(f"[Sched] Round-Robin -> {best_worker}")
        
        new_msg = [best_worker, b"", msgpack.packb(payload)]
        await self.socket.send_multipart(new_msg)