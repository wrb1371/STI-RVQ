"""Independent NumPy packed-layout decoder for sampled visual positions."""
import math
import numpy as np
import torch


def butterfly_numpy(x):
    y=x.astype(np.float32).copy();D=y.shape[-1];stride=1
    while stride<D:
        z=y.reshape(-1,D//(2*stride),2,stride)
        a,b=z[:,:,0].copy(),z[:,:,1].copy()
        z[:,:,0]=a+b;z[:,:,1]=a-b
        stride*=2
    return y*np.float32(1/math.sqrt(D))


@torch.inference_mode()
def verify_layer(codec,l,seed):
    if codec.plan['hybrid_dense']:return {'layer':l,'native_dense':True}
    st=codec.states[l];H,D,M,P,k,W=codec.H,codec.D,codec.M,codec.P,codec.k,codec.W
    raw={key:v.detach().cpu()for key,v in st.items()if torch.is_tensor(v)}
    actual=codec.decode(l,return_pre=True)
    constants={key:v.detach().cpu().numpy()for key,v in codec.constants.items()}
    rng=np.random.default_rng(seed+7919*l);samples=np.unique(np.r_[rng.choice(M,min(128,M),replace=False),np.arange(max(0,P-4),min(M,P+4))])
    total=0
    for side_idx,s in enumerate(['K','V']):
        for h in range(H):
            anchors=raw[s+'_anchor'][h].float().numpy()
            points=np.unique(np.r_[samples,raw['anchor_positions'][h].numpy(),np.arange(P,M)])
            original_positions=codec.maps['video'][torch.tensor(points,device='cuda')].long()
            expected=[]
            for where in points:
                if where>=P:
                    expected.append(anchors[k-W+where-P].copy());continue
                ix=int(raw[s+'_assignment'][h,where]);co=np.float32(raw[s+'_coefficient'][h,where].item())
                pred=anchors[ix]*co
                word,bit=divmod(int(where),64);bits=int(raw[s+'_mask'][h,word])&((1<<64)-1)
                if (bits>>bit)&1:
                    slot=int(raw[s+'_offsets'][h])+int(raw[s+'_prefix'][h,word])+(bits&((1<<bit)-1)).bit_count()
                    packed=raw[s+'_codes'][slot].numpy()
                    ids=np.array([(int(b)>>shift)&3 for b in packed for shift in [0,2,4,6]],dtype=np.int64)
                    rotated=constants['levels'][ids]*np.float32(raw[s+'_scales'][slot].item())
                    correction=butterfly_numpy(rotated[None])[0]*constants['signs'].astype(np.float32)
                    pred=pred+correction
                    if s=='V':assert int(raw['V_chunk_offsets'][slot])==bit
                # Exact anchors overwrite their projection reconstruction.
                anchor_hit=np.flatnonzero(raw['anchor_positions'][h].numpy()==where)
                if len(anchor_hit):pred=anchors[int(anchor_hit[0])].copy()
                expected.append(pred)
            expected=torch.from_numpy(np.stack(expected)).bfloat16()
            observed=actual[side_idx][h,original_positions].cpu()
            assert torch.equal(expected,observed),('independent_decoder_mismatch',l,s,h,int((expected!=observed).sum()),float((expected.float()-observed.float()).abs().max()))
            total+=len(points)
    return {'layer':l,'independent_numpy_BF16_exact':True,'vector_pairs_checked':total,'position_components_checked':total*D}
