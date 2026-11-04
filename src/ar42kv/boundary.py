"""High-pressure catalog: two-stage floor without reference-plan dependence.

Copied from the frozen catalog construction with only its floor-resolution
block replaced. No historical sources or experiment records are overwritten.
"""
import math
import time
import torch
from r15kv.persistent_registry import Arena
from ar22kv.planner import allocate
from ar22kv.joint import JointTemporal
from ar25kv.simple_v1 import ConditionalBudget

class _DirectTwoStage(ConditionalBudget):
    def make_plan(self, meta):
        if self._hybrid_dense:
            return SaturatingInnovations.make_plan(self, meta)
        started = time.perf_counter()
        m, p = len(self._video_list), self._h*self._w
        self._end = self._video_list[-1]+1
        q, text = meta.capacity-self._end, len(self._text_list)
        dtype = torch.int16 if max(meta.position_max) <= 32767 else torch.int32
        global_bytes = 3*meta.capacity*torch.empty((), dtype=dtype).element_size()+256+4*m+4*text
        fixed = self.static_bytes+global_bytes+36*(2*meta.capacity+1024*text)
        rank = self.policy.get('ranking', 'norm')
        variable_budget = self.budget_bytes-fixed
        pressure = 36864*(m+q)/max(1, variable_budget)
        margin = 0
        floors = [(2, 2)]*36
        parent = None
        best = None
        for gop in self.policy.get('gop_choices', [2, 4, 8, 16]):
            anchors = math.ceil(self._nt/gop)*p
            targets = m-anchors
            curves = self.conditional['groups'][self._request_group][rank][str(gop)]
            layer_options = []
            for layer in range(36):
                options = []
                for fi, fraction in enumerate(self.conditional['fractions']):
                    innovations = math.ceil(targets*fraction)
                    slots = anchors+innovations+q
                    for rk in range(floors[layer][0], 7):
                        for rv in range(floors[layer][1], 7):
                            loss = 0.
                            for side, depth in [('K', rk), ('V', rv)]:
                                c = curves[side][layer]
                                factor = self._content_factors[str(gop)][side] if self._content_factors else 1.
                                local = anchors*c['anchor'][depth-1]+targets*factor*c['target'][fi][depth-1]+q*c['query'][depth-1]
                                weight = self._response_weights[side][layer] if self._response_weights and self.policy.get('response_weighted_objective', True) else 1.
                                loss += weight*local
                            options.append({'bytes': slots*(16*(rk+rv)+8), 'loss': loss,
                                            'rK': rk, 'rV': rv, 'innovation_slots': innovations,
                                            'slots': slots, 'fraction_index': fi})
                layer_options.append(options)
            try:
                if self.policy.get('uniform_layers', False):
                    choices = []
                    for oi in range(len(layer_options[0])):
                        same = [layer[oi] for layer in layer_options]
                        cost = sum(x['bytes'] for x in same)
                        if cost <= self.budget_bytes-fixed:
                            choices.append((sum(x['loss'] for x in same), cost, oi, same))
                    if not choices:
                        raise RuntimeError('INFEASIBLE_BY_CONSERVATIVE_JOINT_OPTION_DP')
                    objective, cost, _, selected = min(choices, key=lambda x: x[:3])
                    report = {'objective': objective, 'actual_variable_bytes': cost,
                              'scope': 'complete finite uniform-across-layer joint enumeration'}
                else:
                    selected, report = allocate(layer_options, self.budget_bytes-fixed,
                                                quantum=int(self.policy.get('quantum_bytes', 4096)))
            except RuntimeError as exc:
                if 'INFEASIBLE' not in str(exc):
                    raise
                continue
            key = (report['objective'], fixed+report['actual_variable_bytes'], gop)
            if best is None or key < best[0]:
                best = (key, gop, selected, report)
        if best is None:
            raise RuntimeError('INFEASIBLE_BY_JOINT_PHYSICAL_OPTION_ORACLE')
        _, gop, selected, report = best
        self.policy['gop'] = gop
        self._anchor_ord = [i for i in range(m) if (i//p) % gop == 0]
        anchors = len(self._anchor_ord)
        descriptions = []
        for choice in selected:
            e = choice['slots']
            desc = [('K_index', (2, e, choice['rK'], 8), torch.uint8, 'payload'),
                    ('V_index', (2, e, choice['rV'], 8), torch.uint8, 'payload')]
            desc += [(side+'_norm', (2, e, 1), torch.bfloat16, 'metadata') for side in ['K', 'V']]
            desc += [('T_code', (2, meta.capacity), torch.uint8, 'metadata')]
            desc += [(side, (2, text, 128), torch.bfloat16, 'payload') for side in ['text_FP_K', 'text_FP_V']]
            descriptions.append(desc)
        actual = self.static_bytes+global_bytes+sum(Arena.bytes(x) for x in descriptions)
        assert actual == fixed+report['actual_variable_bytes'] <= self.budget_bytes
        return {'description': descriptions[0], 'descriptions': descriptions,
                'capacity': meta.capacity, 'R_K': selected[0]['rK'], 'R_V': selected[0]['rV'],
                'layer_R_K': [x['rK'] for x in selected], 'layer_R_V': [x['rV'] for x in selected],
                'position_dtype': str(dtype), 'group_dim': 16, 'groups': 8,
                'anchor_tokens': anchors, 'innovation_slots': selected[0]['innovation_slots'],
                'layer_innovation_slots': [x['innovation_slots'] for x in selected],
                'encoded_slots_per_head': selected[0]['slots'],
                'layer_encoded_slots_per_head': [x['slots'] for x in selected],
                'query_slots': q, 'media_original': m, 'protected_prefix_text_tokens': text,
                'gop': gop, 'ranking': rank, 'merged_video_grid': [self._nt, self._h, self._w],
                'estimate': {'codec_static_bytes': self.static_bytes, 'global_request_bytes': global_bytes,
                             'layer_arena_bytes': Arena.bytes(descriptions[0]),
                             'all_layer_arena_bytes': sum(Arena.bytes(x) for x in descriptions),
                             'resident_capacity_bytes': actual, 'kind': 'INTEGER_ESTIMATE_NOT_MEASUREMENT'},
                'conditional_proxy_SHA256': self.conditional_sha, 'joint_allocation': report,
                'ar24_response_weight_SHA256': self._response_sha,
                'ar24_pressure_ratio': pressure, 'ar24_effective_guard_margin': margin,
                'ar24_precision_risk_control': 'direct two-stage floor without a reference plan',
                'ar24_low_risk_min_depth': self.policy.get('low_risk_min_depth', 1),
                'planning_CPU_ms': (time.perf_counter()-started)*1000,
                'guarded_parent_depth': bool(self.policy.get('guarded_parent_depth', False)),
                'content_adaptive': self._content_observation is not None,
                'content_observation': self._content_observation,
                'representation': 'all sampled native positions; one-hop decoded-anchor innovations; joint per-layer rates and occupancy',
                'reference_rate_independent': True, 'hybrid_dense': False}


class BoundaryConditionalBudget(_DirectTwoStage):
    def make_plan(self, meta):
        if self._hybrid_dense:
            return JointTemporal.make_plan(self, meta)
        m = len(self._video_list)
        q = meta.capacity - (self._video_list[-1] + 1)
        t = len(self._text_list)
        bpos = 2 if max(meta.position_max) <= 32767 else 4
        fixed = (self.static_bytes + 3*meta.capacity*bpos + 256 +
                 4*m + 4*t + 36*(2*meta.capacity + 1024*t))
        if self.budget_bytes <= fixed:
            raise RuntimeError('INFEASIBLE_FIXED_COST')
        pressure = 36864*(m+q)/(self.budget_bytes-fixed)
        if pressure <= 32:
            plan = JointTemporal.make_plan(self, meta)
            branch = 'reference_floor_with_one_stage_fallback'
        else:
            plan = _DirectTwoStage.make_plan(self, meta)
            branch = 'direct_two_stage_floor_no_reference'
        plan.update(ar24_pressure_switch_ratio=pressure,
                    ar24_pressure_switch_branch=branch,
                    ar24_pressure_switch_threshold=32.,
                    boundary_contract_version='AR42')
        return plan


from ar26kv.codec import Codec as _VLCodec

class BoundaryVLCodec(_VLCodec):
    def plan_sti(self, teacher):
        L, H, D = 28, 4, 128
        fixed = self.registry.end + L*(H*self.capacity + 4*H*D*self.T)
        if self.budget <= fixed:
            raise RuntimeError('INFEASIBLE_FIXED_COST')
        return super().plan_sti(teacher)
