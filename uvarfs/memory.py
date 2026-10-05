from __future__ import annotations
import math
import numpy as np
import torch

class Reservoir:
    def __init__(self, capacity: int, seed: int = 42):
        self.capacity=capacity; self.rng=np.random.default_rng(seed); self.seen=0; self.arr=None
    def add(self, x: torch.Tensor):
        x=x.detach().float().cpu().numpy()
        if self.arr is None:
            self.arr=np.empty((self.capacity,x.shape[1]),dtype=np.float32)
        for row in x:
            if self.seen < self.capacity: self.arr[self.seen]=row
            else:
                j=int(self.rng.integers(0,self.seen+1))
                if j < self.capacity: self.arr[j]=row
            self.seen += 1
    def value(self):
        n=min(self.seen,self.capacity)
        return self.arr[:n].copy() if self.arr is not None else np.empty((0,0),np.float32)

class CosineIndex:
    def __init__(self, memory: np.ndarray, use_gpu: bool = True, exact: bool = False, nprobe: int = 16):
        import faiss
        if memory.ndim != 2 or len(memory)==0: raise ValueError("empty memory bank")
        self.faiss=faiss; mem=memory.astype(np.float32,copy=True); faiss.normalize_L2(mem)
        d=mem.shape[1]
        if exact or len(mem)<2048:
            index=faiss.IndexFlatIP(d)
        else:
            nlist=max(16,min(1024,int(4*math.sqrt(len(mem)))))
            quant=faiss.IndexFlatIP(d)
            index=faiss.IndexIVFFlat(quant,d,nlist,faiss.METRIC_INNER_PRODUCT)
            index.train(mem)
            index.nprobe=min(nprobe,nlist)
        self.on_gpu=False
        if use_gpu and torch.cuda.is_available() and hasattr(faiss,"StandardGpuResources"):
            try:
                self._res=faiss.StandardGpuResources(); index=faiss.index_cpu_to_gpu(self._res,0,index); self.on_gpu=True
            except Exception: pass
        index.add(mem); self.index=index
    def score(self,q:torch.Tensor)->np.ndarray:
        arr=q.detach().float().cpu().numpy().astype(np.float32,copy=False); self.faiss.normalize_L2(arr)
        sim,_=self.index.search(arr,1); return 1.0-sim[:,0]
