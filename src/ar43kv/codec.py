"""Exact anchors, projection coefficients and packed selective 2-bit residuals.

Independent video port of arXiv:2608.02901, not the authors' implementation.
This quality adapter materializes one layer before native attention; it makes
no claim to reproduce the paper's tiled-attention runtime.
"""
import hashlib,math,time
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from r15kv.persistent_registry import Arena,PersistentRegistry
from ar26kv.codec import apply_rope
from .quantization import LEVELS,encode,decode,fwht,pack_mask,sparse_addresses


def digest(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def layer_description(H,D,M,T,Q,k,W,nK,nV):
    P=M-W;words=(P+63)//64
    z=[('K_anchor',(H,k,D),torch.bfloat16,'payload'),('V_anchor',(H,k,D),torch.bfloat16,'payload'),
       ('anchor_positions',(H,k-W),torch.int64,'metadata')]
    for s in ['K','V']:
        z.extend([(s+'_assignment',(H,P),torch.int16,'metadata'),(s+'_coefficient',(H,P),torch.float16,'metadata'),
                  (s+'_mask',(H,words),torch.int64,'metadata'),(s+'_prefix',(H,words),torch.int32,'metadata'),
                  (s+'_offsets',(H+1,),torch.int32,'metadata')])
    for s,n in [('K',nK),('V',nV)]:
        z.extend([(s+'_codes',(n,D//4),torch.uint8,'payload'),(s+'_scales',(n,),torch.float32,'metadata')])
        if s=='V':z.append(('V_chunk_offsets',(n,),torch.uint8,'metadata'))
    for s in ['K','V']:
        z.extend([('text_'+s,(H,T,D),torch.bfloat16,'payload'),('query_'+s,(H,Q,D),torch.bfloat16,'payload')])
    return z


def causal_statistics(q,kpost,v,query_positions,visual):
    # q: [physical heads, grouped observation queries, D].
    logits=q.float()@kpost.float().transpose(-1,-2)/math.sqrt(q.shape[-1])
    future=torch.arange(kpost.shape[1],device=q.device)[None,:]>query_positions[:,None]
    logits.masked_fill_(future[None],float('-inf'))
    attention=logits.softmax(-1)
    output=attention@v.float()
    visual_attention=attention[:,:,visual]
    score=F.avg_pool1d(visual_attention.mean(1)[:,None],7,stride=1,padding=3)[:,0]
    return visual_attention,output,score


def choose_anchors(score,k,W,seed,layer):
    H,M=score.shape;early=k-W;scored=math.floor(.7*early);out=[]
    for h in range(H):
        ranked=torch.argsort(score[h,:M-W],descending=True,stable=True)
        top=ranked[:scored]
        available=torch.ones(M-W,dtype=torch.bool,device=score.device);available[top]=False
        candidates=available.nonzero().flatten()
        generator=torch.Generator(device=score.device);generator.manual_seed((seed+100003*layer+1009*h)%2**63)
        picked=candidates[torch.randperm(len(candidates),generator=generator,device=score.device)[:early-scored]]
        prefix=torch.cat((top,picked)).sort().values
        out.append(torch.cat((prefix,torch.arange(M-W,M,device=score.device))))
    return torch.stack(out)


def project(x,anchors,positions,P):
    # Coefficients are fitted in FP32 and stored FP16, as two-byte coefficients.
    dot=x.float()@anchors.float().transpose(-1,-2)
    anorm=anchors.float().square().sum(-1).clamp_min(1e-20)
    assignment=(dot.abs()/anorm.sqrt()[:,None]).argmax(-1)
    coefficient=dot.gather(-1,assignment[...,None])[...,0]/anorm.gather(1,assignment)
    for h in range(x.shape[0]):
        early=positions[h][positions[h]<P]
        assignment[h,early]=torch.arange(len(early),device=x.device)
        coefficient[h,early]=1.
    selected=anchors.float().gather(1,assignment[...,None].expand(-1,-1,x.shape[-1]))
    residual=x.float()-coefficient[...,None]*selected
    return assignment[:,:P].to(torch.int16),coefficient[:,:P].half(),residual


def rotate_linear(x,positions,freq,scale):
    phase=positions.float()[:,:,None]*freq[None,None,:]
    widths=[16,24,24];values=[]
    for trig in [torch.cos,torch.sin]:
        z=(trig(phase)*scale).bfloat16().float()
        z=torch.cat([z[i,:,sum(widths[:i]):sum(widths[:i+1])]for i in range(3)],-1)
        values.append(torch.cat((z,z),-1)[None])
    return x*values[0]+torch.cat((-x[...,64:],x[...,:64]),-1)*values[1]


class ReadCache(DynamicCache):
    def __init__(self,codec):
        super().__init__();self.codec=codec;self._seen_tokens=codec.end
    def get_seq_length(self,layer_idx=0):return self.codec.end
    def update(self,k,v,layer_idx,cache_kwargs=None):
        a,b=self.codec.decode(layer_idx)
        return torch.cat((a[None],k),2),torch.cat((b[None],v),2)


class Codec:
    @torch.inference_mode()
    def __init__(self,model,kind,item,pos,vi,ti,end,teacher,seed):
        started=time.time();self.name='AnchorKV-residual';self.kind=kind
        self.L,self.N,self.H,self.D=teacher.geo;L,N,H,D=teacher.geo;assert D==128
        self.end=end;self.M=M=len(vi);self.T=T=len(ti);self.Q=Q=int(item['query_reservation_tokens']);self.capacity=end+Q
        assert M+T==end
        self.budget=int(round(item['budget_mib']*2**20));self.registry=PersistentRegistry(self.budget);self.states={}
        self.plan={'method':self.name,'implementation':'independent paper-based finite-video port','geometry':[L,N,H,D],
                   'media_original':M,'protected_text':T,'query_slots':Q,'budget_bytes':self.budget,'seed':seed,
                   'question_or_answer_used':False,'W':min(32,M),'k':min(M,max(min(32,M),M//128)),
                   'normalization':'absmax + outer-normalized Gaussian levels; convention set on unlabeled WorldSense','layers':[]}
        if L*4*H*D*self.capacity<=self.budget:
            self.plan['hybrid_dense']=True;desc=[(s,(H,self.capacity,D),torch.bfloat16,'payload')for s in ['K','V']]
            for l,raw in teacher.states.items():
                arena=Arena(desc,'cuda',self.registry,f'cache/layer{l}');arena.tensor.zero_();st={'arena':arena,**arena.views}
                st['K'][:,:end].copy_(raw['K_post'].cuda());st['V'][:,:end].copy_(raw['V'].cuda());self.states[l]=st
            self.plan['estimated_resident_bytes']=L*Arena.bytes(desc)
        else:
            self.plan['hybrid_dense']=False;self.W=W=self.plan['W'];self.k=k=self.plan['k'];self.P=P=M-W
            language=model.thinker.model if kind=='Omni'else model.model.language_model
            rotary=language.rotary_emb;self.scale=float(rotary.attention_scaling)
            gen=torch.Generator(device='cpu');gen.manual_seed(seed)
            signs=(torch.randint(0,2,(D,),generator=gen,dtype=torch.int8)*2-1).cuda()
            self.constants={'levels':torch.tensor(LEVELS,device='cuda',dtype=torch.float32),'signs':signs,
                            'freq':rotary.inv_freq.detach().float().clone()};self.registry.own('codec/constants',self.constants,'codec')
            pp=pos[:,:end]if kind=='Omni'else pos[:,0,:end]
            dtype=torch.int16 if int(pp.max())<=32767 else torch.int32
            coords=torch.zeros((3,self.capacity),device='cuda',dtype=dtype);coords[:,:end].copy_(pp)
            self.maps={'positions':coords,'video':vi.to(torch.int32).clone(),'text':ti.to(torch.int32).clone()};self.registry.own('cache/maps',self.maps)
            layer_cap=(self.budget-self.registry.end)//L
            maxn=H*(M-k);lo,hi=0,2*maxn
            if Arena.bytes(layer_description(H,D,M,T,Q,k,W,0,0))>layer_cap:
                raise RuntimeError('INFEASIBLE_ANCHOR_RESIDUAL_FIXED_FORMAT')
            while lo<hi:
                mid=(lo+hi+1)//2
                size=Arena.bytes(layer_description(H,D,M,T,Q,k,W,mid//2,mid-mid//2))
                if size<=layer_cap:lo=mid
                else:hi=mid-1
            self.nK,self.nV=lo//2,lo-lo//2
            desc=layer_description(H,D,M,T,Q,k,W,self.nK,self.nV)
            self.plan.update(nK_per_layer=self.nK,nV_per_layer=self.nV,layer_capacity_bytes=layer_cap,
                             layer_arena_bytes=Arena.bytes(desc),global_and_codec_bytes=self.registry.end,
                             estimated_resident_bytes=self.registry.end+L*Arena.bytes(desc),position_dtype=str(dtype))
            for l,raw in teacher.states.items():
                arena=Arena(desc,'cuda',self.registry,f'cache/layer{l}');arena.tensor.zero_();st={'arena':arena,**arena.views};self.states[l]=st
                self.encode_layer(l,raw,teacher.queries[l],vi,ti,seed)
        torch.cuda.synchronize();self.plan['encoding_seconds']=time.time()-started;self.receipt=self.registry.receipt()
        assert self.receipt['resident_bytes']==self.plan['estimated_resident_bytes']<=self.budget

    @torch.inference_mode()
    def encode_layer(self,l,raw,query,vi,ti,seed):
        H,N,D=self.H,self.N,self.D;M,P,W,k=self.M,self.P,self.W,self.k;st=self.states[l]
        source={s:raw[key].cuda()for s,key in [('K','K_pre'),('V','V')]};kp=raw['K_post'].cuda()
        q=query.cuda().reshape(H,N//H,W,D).flatten(1,2)
        positions=torch.arange(self.end-W,self.end,device='cuda').repeat(N//H)
        attention,output,score=causal_statistics(q,kp,source['V'],positions,vi)
        apos=choose_anchors(score,k,W,seed,l);st['anchor_positions'].copy_(apos[:,:k-W])
        residual={};anchor_mask=torch.zeros((H,M),device='cuda',dtype=torch.bool);anchor_mask.scatter_(1,apos,True)
        for s in ['K','V']:
            x=source[s][:,vi];anchors=x.gather(1,apos[...,None].expand(-1,-1,D));st[s+'_anchor'].copy_(anchors)
            assignment,coefficient,res=project(x,anchors,apos,P)
            st[s+'_assignment'].copy_(assignment);st[s+'_coefficient'].copy_(coefficient);residual[s]=res
            assert torch.isfinite(coefficient).all()
        vr=source['V'][:,vi].float();distance=(vr.square().sum(-1)[:,None]+output.square().sum(-1)[...,None]-2*(output@vr.transpose(-1,-2))).clamp_min_(0)
        rpost=rotate_linear(residual['K'],self.maps['positions'][:,vi].long(),self.constants['freq'],self.scale)
        keydot=q.float()@rpost.transpose(-1,-2)
        utilities={'K':(attention.square()*keydot.square()/D*distance).mean(1),
                   'V':attention.square().mean(1)*residual['V'].square().sum(-1)}
        layer={'layer':l,'anchor_positions_SHA256':digest(apos),'anchor_positions_exact':True}
        for s,n in [('K',self.nK),('V',self.nV)]:
            utility=utilities[s][:,:P].clone();utility[anchor_mask[:,:P]]=float('-inf')
            selected=torch.argsort(utility.flatten(),descending=True,stable=True)[:n].sort().values
            head,where=selected//P,selected%P
            mask=torch.zeros((H,P),device='cuda',dtype=torch.bool);mask[head,where]=True
            packed,prefix,offsets=pack_mask(mask)
            st[s+'_mask'].copy_(packed);st[s+'_prefix'].copy_(prefix);st[s+'_offsets'].copy_(offsets)
            codes,scales=encode(residual[s][head,where],self.constants['signs'],self.constants['levels'])
            st[s+'_codes'].copy_(codes);st[s+'_scales'].copy_(scales)
            if s=='V':st['V_chunk_offsets'].copy_((where%64).to(torch.uint8))
            assert int(mask.sum())==n and not mask[anchor_mask[:,:P]].any()
            hh,ww,slots=sparse_addresses(packed,prefix,offsets,P)
            assert torch.equal(hh,head)and torch.equal(ww,where)and torch.equal(slots,torch.arange(n,device='cuda'))
            layer[s+'_mask_SHA256']=digest(packed);layer[s+'_head_counts']=(offsets[1:]-offsets[:-1]).tolist()
            layer[s+'_codes_SHA256']=digest(codes);layer[s+'_finite_scales']=bool(torch.isfinite(scales).all())
            st['text_'+s].copy_((kp if s=='K'else source['V'])[:,ti])
        # Check the exactness of all anchors using the actual stored decoder.
        decoded=self.decode(l)
        for h in range(H):
            native=vi[apos[h]]
            assert torch.equal(decoded[0][h,native],kp[h,native]),('anchor_K_not_exact',l,h)
            assert torch.equal(decoded[1][h,native],source['V'][h,native]),('anchor_V_not_exact',l,h)
        assert torch.equal(decoded[0][:,ti],kp[:,ti])and torch.equal(decoded[1][:,ti],source['V'][:,ti])
        self.plan['layers'].append(layer)

    @torch.inference_mode()
    def decode(self,l,return_pre=False):
        st=self.states[l];H,D,P,W=self.H,self.D,getattr(self,'P',0),getattr(self,'W',0)
        if self.plan['hybrid_dense']:return st['K'][:,:self.end],st['V'][:,:self.end]
        vi,ti=self.maps['video'].long(),self.maps['text'].long();result={}
        for s in ['K','V']:
            ix=st[s+'_assignment'].long();base=st[s+'_anchor'].float().gather(1,ix[...,None].expand(-1,-1,D))*st[s+'_coefficient'].float()[...,None]
            heads,where,slot=sparse_addresses(st[s+'_mask'],st[s+'_prefix'],st[s+'_offsets'],P)
            if s=='V':where=(where//64)*64+st['V_chunk_offsets'][slot].long()
            correction=decode(st[s+'_codes'],st[s+'_scales'],self.constants['signs'],self.constants['levels'])
            base[heads,where]+=correction[slot]
            out=torch.empty((H,self.end,D),device=vi.device,dtype=torch.bfloat16);out[:,vi[:P]]=base.bfloat16();out[:,vi[P:]]=st[s+'_anchor'][:,-W:]
            for h in range(H):out[h,vi[st['anchor_positions'][h]]]=st[s+'_anchor'][h,:self.k-W]
            if s=='K'and not return_pre:out=apply_rope(out,self.maps['positions'][:,:self.end],self.constants['freq'],self.scale)
            out[:,ti]=st['text_'+s];result[s]=out
        return result['K'],result['V']

    def cache(self):return ReadCache(self)


def preflight(geometry,M,T,Q,position_width,B):
    """Exact aligned minimal admission oracle, independent of K/V and QA labels."""
    L,N,H,D=geometry;W=min(32,M);k=min(M,max(W,M//128));capacity=M+T+Q
    dense=L*4*H*D*capacity
    if dense<=B:return {'feasible':True,'hybrid_dense':True,'minimal_bytes':dense,'budget_bytes':B}
    constants=4*4+D+4*(D//2)
    maps=3*capacity*position_width+4*M+4*T
    desc=layer_description(H,D,M,T,Q,k,W,0,0)
    minimum=constants+maps+L*Arena.bytes(desc)
    return {'feasible':minimum<=B,'hybrid_dense':False,'minimal_bytes':minimum,'budget_bytes':B,
            'M':M,'T':T,'Q':Q,'k':k,'W':W,'constants_bytes':constants,'map_bytes':maps,'minimal_layer_bytes':Arena.bytes(desc)}
