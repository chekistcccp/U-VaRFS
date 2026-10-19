"""Projection oracles for large finite proposals; no scientific budget change."""
import ast,subprocess
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
import pytest
import torch
from uvarfs.performance_uvarfs import project_bounded_simplex

def decimal_projection(values,floor=1e-4,cap=4.):
    # Independent high-precision scalar bisection over the full input range.
    with localcontext() as c:
        c.prec=120
        v=[Decimal.from_float(float(x)) for x in values]
        lower=Decimal.from_float(floor)/len(v)
        upper=min(Decimal(1),Decimal.from_float(cap)/len(v))
        a=min(v)-upper;b=max(v)-lower
        for _ in range(400):
            t=(a+b)/2;s=sum(min(upper,max(lower,x-t)) for x in v)
            if s>1:a=t
            else:b=t
        t=(a+b)/2
        return np.array([float(min(upper,max(lower,x-t))) for x in v])

def legacy():
    root=Path(__file__).resolve().parents[1]
    text=subprocess.check_output(['git','show','3d2bbcefadf946c85793c44e8cd6894daf9d159b:uvarfs/performance_uvarfs.py'],cwd=root,text=True)
    tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='project_bounded_simplex')
    scope={'np':np,'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-projector>','exec'),scope)
    return scope['project_bounded_simplex']

@pytest.mark.parametrize('scale',[1.,1e12,1e20,1e40])
@pytest.mark.parametrize('count',[7,13])
def test_large_range_matches_independent_decimal_oracle(scale,count):
    v=torch.linspace(-1.,1.,count,dtype=torch.float64)*scale
    p=project_bounded_simplex(v)
    assert np.allclose(p,decimal_projection(v),atol=1e-12,rtol=0)
    assert abs(float(p.sum())-1)<1e-12

def test_dense_2304_corner_against_analytic_projection():
    k=2304;v=torch.arange(k,dtype=torch.float64)*1e13
    p=project_bounded_simplex(v);lower=1e-4/k;upper=4/k;capacity=upper-lower
    whole=int(np.floor((1-1e-4)/capacity));remainder=1-1e-4-whole*capacity
    expected=torch.full_like(v,lower);expected[-whole:]=upper;expected[-whole-1]=lower+remainder
    assert torch.allclose(p,expected,atol=1e-12,rtol=0)

def test_ordinary_valid_fast_path_bitwise_unchanged():
    old=legacy();gen=torch.Generator().manual_seed(47)
    for k in [1,7,256,2304]:
        for scale in [0.,1.,100.]:
            v=torch.randn(k,generator=gen,dtype=torch.float64)*scale
            assert torch.equal(old(v),project_bounded_simplex(v))

def test_duplicate_large_coordinates_against_decimal_oracle():
    v=torch.tensor([1e30,1e30,1e30,0.,-1e30,-1e30,-1e30],dtype=torch.float64)
    assert np.allclose(project_bounded_simplex(v),decimal_projection(v),atol=1e-12,rtol=0)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA required')
def test_large_range_cpu_cuda_agree():
    v=torch.arange(2304,dtype=torch.float64)*1e13
    assert torch.allclose(project_bounded_simplex(v),project_bounded_simplex(v.cuda()).cpu(),atol=1e-12,rtol=0)
