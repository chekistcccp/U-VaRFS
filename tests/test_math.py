import numpy as np
import torch
from uvarfs.asls import fit_asls
from uvarfs.u_varfs import fit_uvarfs, apply_uvarfs
from uvarfs.metrics import safe_auc, PixelAccumulator


def test_asls_returns_sparse_layers():
    torch.manual_seed(1)
    layer={i:torch.randn(48,16) for i in range(1,13)}
    var={i:i/12 for i in layer}
    r=fit_asls(layer,var,{"steps":20,"lr":.05,"variability_weight":.1,"sparsity_weight":.02,"geometry_tolerance":.8,"min_layers":2,"max_layers":5})
    assert 2 <= len(r["selected_layers"]) <= 5


def test_uvarfs_shape():
    torch.manual_seed(2)
    x=torch.randn(512,64); v=torch.rand(4,64)
    r=fit_uvarfs(x,v,{"beta":.001,"lambda_grid":[.001,.0001,.00001,.000001],"geometry_tolerance":.3,"max_iter":100,"tol":1e-5,"min_features":8,"max_features":32,"active_threshold":1e-5})
    z=apply_uvarfs(x,r)
    assert z.shape[0] == x.shape[0]
    assert 8 <= z.shape[1] <= 32


def test_stream_pixel_metrics():
    acc=PixelAccumulator(bins=64,pro_thresholds=20)
    m=np.zeros((16,16),np.uint8); m[4:8,4:8]=1
    s=np.zeros((16,16),np.float32); s[4:8,4:8]=1
    acc.update(m,s); r=acc.finalize()
    assert r["pixel_auroc"] > .9
    assert safe_auc([0,1],[0,1]) == 1.0
