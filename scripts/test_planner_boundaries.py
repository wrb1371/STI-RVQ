"""Exercise real catalog DP on the reviewer's counterexample, without a GPU."""
import json, pathlib, sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
from ar42kv.boundary import BoundaryConditionalBudget
from ar21kv.saturating import SaturatingInnovations
from ar22kv.planner import allocate


def fixture(B):
 c=object.__new__(BoundaryConditionalBudget)
 c._hybrid_dense=False;c._video_list=list(range(64,64+21632));c._text_list=list(range(64))
 c._h=c._w=13;c._nt=128;c._base_gop=8;c._request_group='long';c.static_bytes=237568;c.budget_bytes=B
 c.policy={'gop_choices':[2,4,8,16],'ranking':'norm','quantum_bytes':4096,'response_weighted_objective':False}
 c._content_factors=None;c._content_observation=None;c._response_weights=None;c._response_sha=None
 c.conditional_sha='boundary_test_fixture'
 c.conditional={'fractions':[0.,.125,1.],'groups':{'long':{'norm':{}}}}
 for G in [2,4,8,16]:
  curve={'anchor':[1/(r+1)for r in range(6)],'query':[1/(r+1)for r in range(6)],'target':[[((1-f)+.01)/(r+1)for r in range(6)]for f in [0.,.125,1.]]}
  c.conditional['groups']['long']['norm'][str(G)]={s:[curve]*36 for s in ['K','V']}
 meta=SimpleNamespace(capacity=21632+64+288,position_max=(128,13,13))
 return c,meta


def main():
 c,meta=fixture(10*2**20)
 F=c.static_bytes+3*meta.capacity*2+256+4*21632+4*64+36*(2*meta.capacity+1024*64)
 A8=16*169;I8=(21632-A8+7)//8
 ref=F+36*2*(A8+I8+288)*20
 candidate=F+36*2*(8*169+288)*36
 assert ref>c.budget_bytes>=candidate
 with patch.object(SaturatingInnovations,'make_plan',side_effect=AssertionError('REFERENCE_MUST_NOT_BE_CALLED')):
  plan=c.make_plan(meta)
 assert min(plan['layer_R_K']+plan['layer_R_V'])>=2
 assert plan['estimate']['resident_capacity_bytes']<=c.budget_bytes
 checks={'high_pressure_reference_not_called':True,'counterexample_actual_DP_pass':True,
  'reference_MiB':ref/2**20,'candidate_MiB':candidate/2**20,'rho':36864*(21632+288)/(c.budget_bytes-F),
  'actual_plan_resident_bytes':plan['estimate']['resident_capacity_bytes']}
 c,meta=fixture(F)
 try:c.make_plan(meta)
 except RuntimeError as e:assert str(e)=='INFEASIBLE_FIXED_COST'
 else:raise AssertionError('B<=F accepted')
 checks['B_equal_F_rejected']=True
 # Low-pressure reference failure leaves all depth-one alternatives legal.
 c,meta=fixture(64*2**20);c.policy['guarded_parent_depth']=True
 with patch.object(SaturatingInnovations,'make_plan',side_effect=RuntimeError('INFEASIBLE_REFERENCE')) as ref:
  plan=c.make_plan(meta)
 assert ref.call_count==1 and plan['guarded_parent_depth']
 checks['low_pressure_reference_failure_fallback_runs']=True

 c,meta=fixture(F+100)
 try:c.make_plan(meta)
 except RuntimeError as e:assert 'INFEASIBLE' in str(e)
 else:raise AssertionError('impossible catalog accepted')
 checks['all_candidates_infeasible_rejected']=True
 # Exact feasible, but conservative rounding makes two 4097-byte layers
 # require four 4096-byte units, exceeding the three available units.
 opts=[[{'bytes':4097,'loss':1.}]]*2
 assert sum(x[0]['bytes']for x in opts)<=12288
 try:allocate(opts,12288,quantum=4096)
 except RuntimeError as e:assert 'INFEASIBLE' in str(e)
 else:raise AssertionError('rounded infeasible case not reported')
 checks['rounded_infeasible_distinguished_from_exact_feasibility']=True

 print('Boundary tests passed (CPU only).')
if __name__=='__main__':main()
