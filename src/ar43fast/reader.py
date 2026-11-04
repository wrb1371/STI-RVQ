"""Fused 2-bit decode, Walsh-Hadamard inverse, projection and native MRoPE.

The reference codec and layout are unchanged. Every constructed layer is checked
against the frozen Torch reader before any native QA uses this implementation.
Rotary tables live in a per-question workspace, not in persistent codec state.
"""
import math
import torch
import triton
import triton.language as tl
from ar43kv.codec import Codec as ReferenceCodec,ReadCache as ReferenceCache

@triton.jit
def _butterfly(z,S:tl.constexpr,B:tl.constexpr):
    shaped=tl.reshape(z,(B,128//(2*S),2,S))
    a,b=tl.split(tl.permute(shaped,(0,1,3,2)))
    combined=tl.join(a+b,a-b)
    return tl.reshape(tl.permute(combined,(0,1,3,2)),(B,128))

@triton.jit(do_not_specialize=['END','M','T','P','K','W','WORDS'])
def _read(Anchor,Assignment,Coefficient,Mask,Prefix,Offsets,Codes,Scales,Chunk,Signs,Levels,TextKV,Vid,Text,Cos,Sin,Out,
          END,M,T,P,K,W,WORDS,SIDE:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B);h=tl.program_id(1);d=tl.arange(0,128)
    valid=row<M+T;visual=row<M;early=visual&(row<P);window=visual&~early;istext=(row>=M)&valid
    aid=tl.load(Assignment+h*P+row,early,0).to(tl.int32)
    aid=tl.where(window,K-W+row-P,aid)
    gain=tl.load(Coefficient+h*P+row,early,1).to(tl.float32)
    anchor=tl.load(Anchor+(h*K+aid[:,None])*128+d[None,:],visual[:,None],0).to(tl.float32)
    base=anchor*gain[:,None]
    word=row//64;bit=row%64
    mask=tl.load(Mask+h*WORDS+word,early,0).to(tl.uint64)
    active=early&(((mask>>bit.to(tl.uint64))&1)!=0)
    lower=mask&((tl.full((B,),1,tl.uint64)<<bit.to(tl.uint64))-1)
    pop=tl.inline_asm_elementwise('popc.b64 $0, $1;',constraints='=r,l',args=[lower],dtype=tl.int32,is_pure=True,pack=1)
    slot=tl.load(Offsets+h)+tl.load(Prefix+h*WORDS+word,early,0)+pop
    byte=tl.load(Codes+slot[:,None]*32+d[None,:]//4,active[:,None],0).to(tl.int32)
    code=(byte>>(2*(d[None,:]%4)))&3
    level=tl.load(Levels+code)
    scale=tl.load(Scales+slot,active,0)
    z=level*scale[:,None]
    z=_butterfly(z,1,B)
    z=_butterfly(z,2,B)
    z=_butterfly(z,4,B)
    z=_butterfly(z,8,B)
    z=_butterfly(z,16,B)
    z=_butterfly(z,32,B)
    z=_butterfly(z,64,B)
    z=z*0.08838834764831845
    signs=tl.load(Signs+d).to(tl.float32)
    residual=z*signs[None,:]
    value=tl.where(active[:,None],base+residual,base).to(tl.bfloat16).to(tl.float32)
    ordinal=row
    if SIDE==1:
        chunk=tl.load(Chunk+slot,active,0).to(tl.int32)
        ordinal=tl.where(active,word*64+chunk,row)
    seq=tl.load(Vid+ordinal,visual,0)
    seq=tl.where(istext,tl.load(Text+row-M,istext,0),seq)
    if SIDE==0:
        first,second=tl.split(tl.permute(tl.reshape(value,(B,2,64)),(0,2,1)))
        rotated=tl.reshape(tl.permute(tl.join(-second,first),(0,2,1)),(B,128))
        axis=tl.where(d%64<16,0,tl.where(d%64<40,1,2))
        at=(axis[None,:]*END+seq[:,None])*64+d[None,:]%64
        co=tl.load(Cos+at,visual[:,None],0).to(tl.float32)
        si=tl.load(Sin+at,visual[:,None],0).to(tl.float32)
        left=(value*co).to(tl.bfloat16).to(tl.float32)
        right=(rotated*si).to(tl.bfloat16).to(tl.float32)
        value=(left+right).to(tl.bfloat16)
    else:value=value.to(tl.bfloat16)
    textvalue=tl.load(TextKV+(h*T+row[:,None]-M)*128+d[None,:],istext[:,None],0)
    value=tl.where(istext[:,None],textvalue,value)
    tl.store(Out+(h*END+seq[:,None])*128+d[None,:],value,valid[:,None])


def workspace(codec):
    phase=codec.maps['positions'][:,:codec.end].float()[:,:,None]*codec.constants['freq'][None,None,:]
    return (phase.cos().mul(codec.scale).bfloat16(),phase.sin().mul(codec.scale).bfloat16())

@torch.inference_mode()
def read(codec,l,ws=None):
    if codec.plan['hybrid_dense']:return ReferenceCodec.decode(codec,l)
    st=codec.states[l];ws=workspace(codec)if ws is None else ws;out=[]
    for side,s in enumerate(['K','V']):
        x=torch.empty((codec.H,codec.end,128),device='cuda',dtype=torch.bfloat16)
        _read[(triton.cdiv(codec.end,4),codec.H)](st[s+'_anchor'],st[s+'_assignment'],st[s+'_coefficient'],st[s+'_mask'],st[s+'_prefix'],st[s+'_offsets'],st[s+'_codes'],st[s+'_scales'],st['V_chunk_offsets'],codec.constants['signs'],codec.constants['levels'],st['text_'+s],codec.maps['video'],codec.maps['text'],*ws,x,
               codec.end,codec.M,codec.T,codec.P,codec.k,codec.W,(codec.P+63)//64,side,4,num_warps=4,enable_fp_fusion=False)
        out.append(x)
    return tuple(out)


class Cache(ReferenceCache):
    def __init__(self,codec):
        super().__init__(codec);self.workspace=None if codec.plan['hybrid_dense']else workspace(codec)
    def update(self,k,v,layer_idx,cache_kwargs=None):
        a,b=self.codec.decode(layer_idx,ws=self.workspace)
        return torch.cat((a[None],k),2),torch.cat((b[None],v),2)


class Codec(ReferenceCodec):
    def __init__(self,*args,**kwargs):
        self._verified_layers=set();super().__init__(*args,**kwargs)
        self.plan['execution_extension']={'reader':'fused_two_bit_FWHT_projection_MRoPE','all_layers_exact_to_frozen_Torch':len(self._verified_layers)==self.L if not self.plan['hybrid_dense']else True,'checked_layers':len(self._verified_layers),'workspace_scope':'one question, released with its read cache','packed_format_unchanged':True}
    @torch.inference_mode()
    def decode(self,l,return_pre=False,ws=None):
        if return_pre or self.plan['hybrid_dense']:return ReferenceCodec.decode(self,l,return_pre)
        got=read(self,l,ws)
        if l not in self._verified_layers:
            reference=ReferenceCodec.decode(self,l)
            for side,a,b in zip(['K','V'],got,reference):
                assert torch.equal(a,b),('fused_reader_not_exact',l,side,int((a!=b).sum()),float((a.float()-b.float()).abs().max()))
            self._verified_layers.add(l)
        return got
    def cache(self):return Cache(self)
