"""Independent numerical and packing checks, before any benchmark quality run."""
import json,pathlib,sys,math
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from ar43kv.quantization import LEVELS,fwht,encode,decode,pack_mask,sparse_addresses
from ar43kv.codec import project,choose_anchors,causal_statistics,preflight,layer_description
from r14kv.fitting import hadamard

def main():
 torch.set_num_threads(2);torch.manual_seed(123)
 D=128;H=hadamard(D).float();x=torch.randn(200,D)
 transformed=fwht(x);assert torch.allclose(transformed,x@H,atol=2e-6,rtol=2e-6)
 assert torch.allclose(fwht(transformed),x,atol=2e-6,rtol=2e-6)
 signs=(torch.randint(0,2,(D,),dtype=torch.int8)*2-1);levels=torch.tensor(LEVELS)
 codes,scales=encode(x,signs,levels);rec=decode(codes,scales,signs,levels)
 # Independent dense inverse: unpack byte positions with Python bit arithmetic.
 rows=np.array([[((int(b)>>s)&3)for b in row for s in [0,2,4,6]]for row in codes.numpy()])
 dense=(np.array(LEVELS)[rows]*scales.numpy()[:,None])@H.numpy().astype(np.float64)
 dense*=signs.numpy()[None]
 assert np.allclose(rec.numpy(),dense,rtol=2e-5,atol=2e-5)
 distortion=float((rec-x).square().sum()/x.square().sum());assert 0<distortion<1
 checks=0
 for heads in [1,2,4]:
  for P in [1,63,64,65,127,128,129,511]:
   mask=torch.rand(heads,P)<.27;packed,prefix,offsets=pack_mask(mask);hh,pp,slot=sparse_addresses(packed,prefix,offsets,P)
   truth=mask.nonzero();assert torch.equal(hh,truth[:,0])and torch.equal(pp,truth[:,1]);assert torch.equal(slot,torch.arange(len(truth)))
   # Explicit unsigned Python bit test, including bit63 and cross-word edges.
   for h in range(heads):
    for i in range(P):assert bool((int(packed[h,i//64])&((1<<64)-1))>>(i%64)&1)==bool(mask[h,i])
   checks+=1
 x=torch.randn(2,180,D);apos=torch.tensor([list(range(0,20))+list(range(148,180))]*2);anchors=x.gather(1,apos[...,None].expand(-1,-1,D))
 assignment,coefficient,res=project(x,anchors,apos,148)
 assert torch.equal(assignment[:,:20],torch.arange(20)[None].expand(2,-1))and torch.equal(coefficient[:,:20],torch.ones((2,20),dtype=torch.float16))
 assert float((res*anchors.gather(1,torch.cat((assignment.long(),torch.arange(20,52)[None].expand(2,-1)),1)[...,None].expand(-1,-1,D))).sum(-1).abs().max())<1e-3
 score=torch.randn(2,5000);a=choose_anchors(score,39,32,42,0);b=choose_anchors(score,39,32,42,0)
 assert torch.equal(a,b)and all(len(torch.unique(r))==39 for r in a)
 assert torch.equal(a[:,-32:],torch.arange(4968,5000)[None].expand(2,-1))
 # Account for BF16 query capacity without silently reducing it to fit.
 one=preflight((36,16,2,128),5184,60,192,2,16*2**20)
 two=preflight((28,28,4,128),5184,14,192,2,16*2**20)
 assert one['feasible']and not two['feasible']
 out={'completed':True,'device':'CPU','packing_shapes_checked':checks,'FWHT_dense_oracle_checked':True,
      'projection_self_assignment_checked':True,'mask_word63_and_boundaries_checked':True,'literal_quantizer_relative_MSE_on_synthetic_Gaussian':distortion,
      'example_preflight_Omni':one,'example_preflight_VL':two,'no_benchmark_labels_or_scores_used':True}
 print('Anchor-residual packing, projection and scalar-codec CPU checks passed.')
if __name__=='__main__':main()
