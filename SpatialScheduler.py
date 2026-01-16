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
        self.worker_history = defaultdict(lambda: deque(maxlen=history_len)) # 记录每个 Worker 最近处理的 N 个 BBox
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
                await self._forward_result(payload)
            else:
                print(f"[Scheduler] Unknown message type: {msg_type}")

    def _handle_worker_ready(self, worker_id):
        if worker_id not in self.workers:
            self.workers.add(worker_id)
            print(f"[Scheduler] Worker registered: {worker_id}")
            self.ready_event.set()

    async def _dispatch_request(self, client_id, payload, raw_frames):
        """
        核心调度逻辑：Stateful Intersection -> Stateless Z-Order
        """

        if not self.workers:
            # 简单处理：如果没有 Worker，阻塞等待或报错
            print("[Scheduler] No workers available! Waiting...")
            await self.ready_event.wait()

        bbox = payload['bbox'] # Flat list [x1,y1,z1, x2,y2,z2]
        
        best_worker = None
        max_intersection = 0
        
        # 策略 A: 寻找最大空间重叠 (Cache Affinity)
        for w_id in self.workers:
            history = self.worker_history[w_id]
            current_overlap = 0
            for past_bbox in history:
                vol = utils.calc_intersection_volume(bbox, past_bbox)
                current_overlap += vol
            
            if current_overlap > max_intersection:
                max_intersection = current_overlap
                best_worker = w_id
        
        # 策略 B: 如果没有重叠，使用 Morton Code (Load Balancing + Spatial Locality)
        if best_worker is None:
            # 取中心点计算 Morton Code
            cx = (bbox[0] + bbox[3]) // 2
            cy = (bbox[1] + bbox[4]) // 2
            cz = (bbox[2] + bbox[5]) // 2
            code = utils.morton_code_3d(cx, cy, cz)
            
            # 一致性哈希简单模拟：取模
            worker_list = sorted(list(self.workers))
            best_worker = worker_list[code % len(worker_list)]
            # print(f"[Scheduler] Strategy B (Z-Order) -> {best_worker}")
        else:
            # print(f"[Scheduler] Strategy A (Overlap: {max_intersection}) -> {best_worker}")
            pass

        # 更新该 Worker 的历史记录
        self.worker_history[best_worker].append(bbox)

        # 转发请求给 Worker
        # Router 发给 Dealer 需要指定 ID: [Worker_ID, Empty, Client_ID, Empty, Payload]
        # 我们需要把 Client ID 塞进 Payload 或者作为信封传过去
        # 这里为了简化，我们修改 Payload，加入 Client ID，让 Worker 处理完后带着 Client ID 回复
        payload['client_id'] = client_id # 注入 Client Identity
        print(f"best_worker={best_worker}")
        new_msg = [best_worker, b"", msgpack.packb(payload)]
        await self.socket.send_multipart(new_msg)

    async def _forward_result(self, payload):
        """
        收到 Worker 结果，转发回 Client
        """
        client_id = payload['client_id']
        # Router 回复 Client: [Client_ID, Empty, Payload]
        # 注意 payload 里已经包含了 result status, req_id 等
        await self.socket.send_multipart([client_id, b"", msgpack.packb(payload)])