from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, auc
from scipy import ndimage


def prepare_pixel_mask(mask):
    """Final-evaluation-only mask data, reusable by every compared method."""
    mask=np.asarray(mask,dtype=bool)
    label,count=ndimage.label(mask) if mask.any() else (np.zeros_like(mask,dtype=np.int32),0)
    flat=label.ravel()
    return {'mask':mask,'normal':~mask,
            'regions':tuple(np.flatnonzero(flat==rid) for rid in range(1,count+1))}


def safe_auc(y,s):
    y=np.asarray(y); s=np.asarray(s)
    return float(roc_auc_score(y,s)) if len(np.unique(y))>1 else float("nan")
def safe_ap(y,s):
    y=np.asarray(y); s=np.asarray(s)
    return float(average_precision_score(y,s)) if len(np.unique(y))>1 else float("nan")
def bootstrap_auc(y,s,n=1000,seed=42,batch_size=32):
    """Original image bootstrap, with one score sort and exact tie handling.

    Keep the same integers-based, non-stratified resampling stream and skip
    single-class draws. Multiplicities replace sorting every resampled score
    array; batching changes only temporary storage, not the draws or CI.
    """
    if n<=0:
        return [float('nan'),float('nan')]
    y=np.asarray(y); s=np.asarray(s)
    if y.ndim!=1 or s.shape!=y.shape or not len(y) or not np.isfinite(s).all() or batch_size<1:
        raise ValueError('bootstrap requires aligned finite scores and a positive batch size')
    classes=np.unique(y)
    if len(classes)<2:
        return [float('nan'),float('nan')]
    if len(classes)!=2:
        raise ValueError('image AUROC bootstrap requires binary labels')
    order=np.argsort(s,kind='stable')
    starts=np.r_[0,np.flatnonzero(s[order][1:]!=s[order][:-1])+1]
    positive=y[order]==classes[-1]
    rng=np.random.default_rng(seed); vals=[]; count=len(y)
    for start in range(0,n,batch_size):
        size=min(batch_size,n-start)
        idx=rng.integers(0,count,(size,count))
        weights=np.bincount((idx+np.arange(size)[:,None]*count).ravel(),
                            minlength=size*count).reshape(size,count)[:,order]
        pos=np.add.reduceat(weights*positive,starts,axis=1)
        neg=np.add.reduceat(weights*~positive,starts,axis=1)
        P=pos.sum(1); N=neg.sum(1); valid=(P>0)&(N>0)
        below=np.cumsum(neg,axis=1)-neg
        numerator=(pos*(below+.5*neg)).sum(1)
        vals.extend((numerator[valid]/(P[valid]*N[valid])).tolist())
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))] if vals else [float('nan'),float('nan')]

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
        return (vals.size-np.searchsorted(vals,thresholds,side='left')).astype(np.float64)

    def _histogram_and_counts(self,values):
        # Sort once for BOTH the pixel histogram and fixed PRO thresholds.
        # The final histogram edge is inclusive, as in numpy.histogram.
        vals=np.sort(values.astype(np.float32,copy=False))
        boundaries=np.searchsorted(vals,self.edges,side='left')
        boundaries[-1]=np.searchsorted(vals,self.edges[-1],side='right')
        counts=(vals.size-np.searchsorted(vals,self.thresholds,side='left')).astype(np.float64)
        return np.diff(boundaries),counts

    def update(self,mask,score,prepared=None):
        prepared=prepare_pixel_mask(mask) if prepared is None else prepared
        mask=prepared['mask']
        score=np.asarray(score,dtype=np.float32)
        if mask.shape!=score.shape:
            raise ValueError('pixel score and mask shapes must match')
        score=np.clip(score,self.edges[0],self.edges[-1]-1e-7)

        positive,_=self._histogram_and_counts(score[mask])
        normal=score[prepared['normal']]
        negative,counts=self._histogram_and_counts(normal)
        self.pos += positive
        self.neg += negative
        self.normal_pixels += normal.size
        if normal.size:
            self.fp += counts

        flat=score.ravel()
        for indices in prepared['regions']:
            vals=flat[indices]
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
