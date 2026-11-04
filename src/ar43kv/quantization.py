"""Absmax-normalized, randomized-Hadamard Gaussian two-bit codec."""
import math
import torch

# Lloyd-Max stationary centroids of a unit Gaussian (four symmetric levels).
GAUSSIAN_LEVELS=(-1.5104176084990954,-0.452780034636492,0.452780034636492,1.5104176084990954)
# A single convention is chosen before the full panel on unlabeled WorldSense.
_literal=torch.tensor(GAUSSIAN_LEVELS,dtype=torch.float32)
LEVELS=tuple((_literal/_literal.abs().max()).tolist())
del _literal

def fwht(x):
    n=x.shape[-1]
    assert n>0 and n&(n-1)==0
    y=x.float()
    if y.numel()==0:return y
    width=1
    while width<n:
        z=y.reshape(*y.shape[:-1],-1,2,width)
        a,b=z[...,0,:],z[...,1,:]
        y=torch.stack((a+b,a-b),-2).reshape_as(y)
        width*=2
    return y*(1/math.sqrt(n))

def encode(x,signs,levels):
    rotated=fwht(x.float()*signs.float())
    scale=rotated.abs().amax(-1)
    unit=rotated/scale.clamp_min(1e-12)[...,None]
    ids=torch.bucketize(unit.contiguous(),(levels[:-1]+levels[1:])/2).to(torch.uint8)
    n=ids.shape[-1];assert n%4==0
    z=ids.reshape(*ids.shape[:-1],n//4,4)
    packed=z[...,0]|(z[...,1]<<2)|(z[...,2]<<4)|(z[...,3]<<6)
    return packed.contiguous(),scale.float().contiguous()

def decode(packed,scale,signs,levels):
    ix=torch.stack([(packed>>shift)&3 for shift in [0,2,4,6]],-1).reshape(*packed.shape[:-1],packed.shape[-1]*4).long()
    rotated=levels[ix]*scale[...,None]
    return fwht(rotated)*signs.float()

def pack_mask(mask):
    H,P=mask.shape;words=(P+63)//64
    padded=torch.zeros((H,words*64),device=mask.device,dtype=torch.bool);padded[:,:P]=mask
    bits=torch.ones(64,device=mask.device,dtype=torch.int64)<<torch.arange(64,device=mask.device)
    block=padded.reshape(H,words,64)
    packed=(block.long()*bits).sum(-1)
    count=block.sum(-1).to(torch.int32)
    prefix=(count.cumsum(-1)-count).to(torch.int32)
    total=count.sum(-1).to(torch.int32)
    offsets=torch.cat((torch.zeros(1,device=mask.device,dtype=torch.int32),total.cumsum(0).to(torch.int32)))
    return packed,prefix,offsets

def sparse_addresses(packed,prefix,offsets,P):
    shifts=torch.arange(64,device=packed.device)
    block=((packed[...,None]>>shifts)&1).bool()
    mask=block.flatten(1)[:,:P]
    head,pos=mask.nonzero(as_tuple=True)
    word,bit=pos//64,pos%64
    within=block.long().cumsum(-1)-1
    slot=offsets[head].long()+prefix[head,word].long()+within[head,word,bit]
    return head,pos,slot
