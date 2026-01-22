import zmq
import msgpack
import numpy as np
import traceback
import os
import sys
import ctypes
import time
from multiprocessing import shared_memory
from utils import _fast_fill

LOCAL_CLONE = '/CX/neuro_tracking/xinr/cloudvolume_test/Cloudvolume'
if os.path.exists(LOCAL_CLONE):
    sys.path.insert(0, LOCAL_CLONE)

from cloudvolume import CloudVolume

class VolumeWorker:
    def __init__(self, scheduler_addr, worker_idx, parallel=4):
        self.scheduler_addr = scheduler_addr
        self.worker_id = f"worker_{worker_idx}_{os.getpid()}".encode('utf-8')
        self.parallel = parallel
        
        print(f"[{self.worker_id.decode()}] Init CloudVolume (Parallel={parallel})...")
        self.cv = CloudVolume(
            '/CX/neuro_tracking/fafb-ffn1', 
            mip=0,
            fill_missing=True,
            log_path="/dev/shm/222.log",
            cache=True,
            lru_bytes=80 * 1024**2, 
            partial_decompress_parallel=parallel 
        )
        self.cv.cache_thread = 0
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.identity = self.worker_id

    def run(self):
        try:
            self.socket.connect(self.scheduler_addr)
            # 发送 READY 时携带自身的并行能力信息
            ready_payload = {"type": "READY", "parallel": self.parallel}
            self.socket.send(msgpack.packb(ready_payload))
            
            while True:
                frames = self.socket.recv_multipart()
                req = msgpack.unpackb(frames[-1])
                self._process_request(req)
        except Exception:
            traceback.print_exc()

    def _process_request(self, req):
        time1=time.perf_counter()
        client_id = req['client_id']
        req_id = req['req_id']
        shm_name = req['shm_name']
        shape = req['shape']
        dtype_str = req['dtype']
        order = req['order']
        bbox_list = req['bbox']
        bg_color = req.get('bg_color', 0) # 获取需要填充的颜色
        data_size = req['data_size']
        try:

            existing_shm=shared_memory.SharedMemory(create=True, size=int(data_size), name=shm_name)
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister(existing_shm._name, 'shared_memory')
            except:
                pass
            time2=time.perf_counter()

            _fast_fill(existing_shm.buf, shape, dtype_str, self.parallel, bg_color, order)
            time3=time.perf_counter()

            slices = [
                slice(bbox_list[0], bbox_list[3]),
                slice(bbox_list[1], bbox_list[4]),
                slice(bbox_list[2], bbox_list[5])
            ]
            
            self.cv.renderbuffer = existing_shm.buf
            self.cv[slices[0], slices[1], slices[2]]
            time4=time.perf_counter()
            print(f"Prepare time {(time2-time1)*1000}ms, _fast_fill time {(time3-time2)*1000}ms, cv time {(time4-time3)*1000}ms")

            existing_shm.close()

            resp = {
                "type": "RESULT",
                "client_id": client_id,
                "req_id": req_id,
                "status": "OK"
            }
            self.socket.send(msgpack.packb(resp))

        except Exception as e:
            err_resp = {
                "type": "RESULT",
                "client_id": client_id,
                "req_id": req_id,
                "status": "ERROR",
                "error": str(e)
            }
            self.socket.send(msgpack.packb(err_resp))
            traceback.print_exc()
