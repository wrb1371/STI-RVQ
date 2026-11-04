"""Question-independent native prefix and last-window query capture."""
import torch
from transformers.cache_utils import DynamicCache
from ar32kv.runtime import geometry,Supplier,answer
from ar37cq.runtime import load,sha_tensor
from ar26kv.runtime import media_kwargs,last_projection

class Teacher:
    def __init__(self,model,kind):
        self.model,self.kind=model,kind;self.language,self.geo=geometry(model,kind)
        self.active=False;self.handles=[];self.states={};self.queries={};self.pre={}
        L,N,H,D=self.geo
        if kind=='Omni':
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
        for l,block in enumerate(self.language.layers):
            def khook(module,args,out,l=l):
                if self.active:self.pre[l]=out.detach().view(1,-1,H,D).transpose(1,2)[0].contiguous().cpu()
            def qhook(module,args,out,l=l):
                if not self.active:return
                q=out.detach().view(1,-1,N,D).transpose(1,2)[:,:,-self.W:]
                q,_=apply_multimodal_rotary_pos_emb(q,q,self.cos[:,:,-self.W:],self.sin[:,:,-self.W:],self.sections)
                self.queries[l]=q[0].contiguous().cpu()
            self.handles.extend((block.self_attn.k_proj.register_forward_hook(khook),block.self_attn.q_proj.register_forward_hook(qhook)))
    @torch.inference_mode()
    def capture(self,x,pos,end,W):
        self.end,self.W=end,W;self.states={};self.queries={};self.pre={}
        pp=pos[:,:end][:,None]if self.kind=='Omni'else pos[:,:,:end]
        self.cos,self.sin=self.language.rotary_emb(torch.empty((1,1,self.geo[-1]),device='cuda',dtype=torch.bfloat16),pp)
        self.sections=self.language.layers[0].self_attn.rope_scaling['mrope_section']
        cache=DynamicCache();self.active=True
        try:
            if self.kind=='VL':
                with last_projection(self.model):self.model(**media_kwargs(x,pos,end),past_key_values=cache)
            else:
                from ar18kv.prefix import media_prefill
                media_prefill(self.model,x,cache,pos,end,False)
        finally:self.active=False
        for l in range(self.geo[0]):
            self.states[l]={'K_pre':self.pre[l],'K_post':cache.key_cache[l][0].contiguous().cpu(),'V':cache.value_cache[l][0].contiguous().cpu()}
        self.cos=self.sin=None;self.pre={};del cache
        self.ids_sha=sha_tensor(x['input_ids'][:,:end].cpu());self.pos_sha=sha_tensor(pp.cpu())
        torch.cuda.synchronize()
    def assert_same(self,x,pos,end):
        pp=pos[:,:end][:,None]if self.kind=='Omni'else pos[:,:,:end]
        assert self.end==end and self.ids_sha==sha_tensor(x['input_ids'][:,:end].cpu())and self.pos_sha==sha_tensor(pp.cpu())
    def bf16_cache(self):
        cache=DynamicCache();cache.key_cache=[s['K_post'][None].cuda()for s in self.states.values()];cache.value_cache=[s['V'][None].cuda()for s in self.states.values()];cache._seen_tokens=self.end;return cache
    def clear(self):self.states={};self.queries={};self.pre={}
    def close(self):
        self.clear()
        for h in self.handles:h.remove()
