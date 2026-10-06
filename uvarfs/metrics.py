from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, auc
from scipy import ndimage


def safe_auc(y,s):
    y=np.asarray(y); s=np.asarray(s)
    return float(roc_auc_score(y,s)) if len(np.unique(y))>1 else float("nan")
def safe_ap(y,s):
    y=np.asarray(y); s=np.asarray(s)
    return float(average_precision_score(y,s)) if len(np.unique(y))>1 else float("nan")
def bootstrap_auc(y,s,n=1000,seed=42):
    y=np.asarray(y); s=np.asarray(s); rng=np.random.default_rng(seed); vals=[]
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))<2: continue
        vals.append(roc_auc_score(y[idx],s[idx]))
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))] if vals else [float("nan"),float("nan")]

class PixelAccumulator:
    """Streaming approximate pixel AUROC/AUPRC/AUPRO with low CPU overhead."""
    def __init__(self,bins=1024,pro_thresholds=100,max_score=2.0,max_fpr=0.3):
        self.bins=bins
        self.edges=np.linspace(0,max_score,bins+1)
        self.pos=np.zeros(bins,np.int64)
        self.neg=np.zeros(bins,np.int64)
        self.thresholds=np.linspace(max_score,0,pro_thresholds)
        self.fp=np.zeros(pro_thresholds,np.float64)
        self.normal_pixels=0
        self.pro=np.zeros(pro_thresholds,np.float64)
        self.regions=0
        self.max_fpr=max_fpr

    @staticmethod
    def _counts_ge(values, thresholds):
        if values.size == 0:
            return np.zeros(len(thresholds),dtype=np.float64)
        vals=np.sort(values.astype(np.float32,copy=False))
        idx=np.searchsorted(vals,thresholds,side='left')
        return (vals.size-idx).astype(np.float64)

    def update(self,mask,score):
        mask=np.asarray(mask).astype(bool)
        score=np.asarray(score,dtype=np.float32)
        score=np.clip(score,self.edges[0],self.edges[-1]-1e-7)

        self.pos += np.histogram(score[mask],bins=self.edges)[0]
        self.neg += np.histogram(score[~mask],bins=self.edges)[0]

        normal=score[~mask]
        self.normal_pixels += normal.size
        if normal.size:
            self.fp += self._counts_ge(normal,self.thresholds)

        lab,n=ndimage.label(mask)
        for rid in range(1,n+1):
            vals=score[lab==rid]
            if vals.size:
                self.pro += self._counts_ge(vals,self.thresholds)/vals.size
                self.regions += 1

    def finalize(self):
        P=int(self.pos.sum())
        N=int(self.neg.sum())
        tp=np.cumsum(self.pos[::-1])
        fp=np.cumsum(self.neg[::-1])
        tpr=tp/max(P,1)
        fpr=fp/max(N,1)
        pixel_auc=float(auc(np.r_[0.0,fpr],np.r_[0.0,tpr])) if P and N else float('nan')
        precision=tp/np.maximum(tp+fp,1)
        recall=tpr
        # Histogram approximation to average precision, including the first
        # recall jump. Trapezoidal PR integration can give 0.5 for perfect scores.
        pixel_ap=float(np.sum(np.diff(np.r_[0.0,recall])*precision)) if P else float('nan')

        if self.regions and self.normal_pixels:
            pro=self.pro/self.regions
            pfpr=self.fp/self.normal_pixels
            order=np.argsort(pfpr)
            x,y=pfpr[order],pro[order]
            ux=np.unique(x)
            uy=np.array([y[x==v].max() for v in ux])
            if len(ux)>1:
                # Interpolate to the exact 0.3 FPR limit rather than dropping
                # the final interval or returning NaN for a perfect map.
                cutoff=np.interp(self.max_fpr,ux,uy)
                keep=ux<self.max_fpr
                xx=np.r_[ux[keep],self.max_fpr]
                yy=np.r_[uy[keep],cutoff]
                aupro=float(auc(xx,yy)/self.max_fpr) if len(xx)>1 else float('nan')
            else:
                aupro=float('nan')
        else:
            aupro=float('nan')
        return {"pixel_auroc":pixel_auc,"pixel_auprc":pixel_ap,"aupro":aupro}
