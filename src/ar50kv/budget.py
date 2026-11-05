"""A codec-independent index lower bound; it never declares feasibility.

The original K2/V2 format keeps an index for every coordinate of both keys and
values. Rotations, local ranges, norms, text, queries, maps and padding can only
increase its cost. This check does not run a model or measure answer quality.
"""


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def visual_index_bytes(layers, kv_heads, head_dim, video_tokens):
    values = [positive_integer(x, n) for x, n in zip(
        [layers, kv_heads, head_dim, video_tokens],
        ['layers', 'kv_heads', 'head_dim', 'video_tokens'])]
    # Independent ceiling per side: two bits for every visual coordinate.
    coordinates = values[0] * values[1] * values[2] * values[3]
    return 2 * ((2 * coordinates + 7) // 8)


def check(record):
    minimum = visual_index_bytes(*(record[n] for n in
        ['layers', 'kv_heads', 'head_dim', 'video_tokens']))
    budget = record.get('budget_bytes')
    if budget is None:
        budget = round(float(record['budget_mib']) * 2**20)
    budget = positive_integer(budget, 'budget_bytes')
    rejected = minimum > budget
    return {
        'id': record.get('id'), 'video': record.get('video'),
        'model': record.get('model'), 'dataset': record.get('dataset'),
        'duration_group': record.get('duration_group'), 'method': 'OmniKVQuant',
        'format': 'K2/V2, all visual positions', 'budget_bytes': budget,
        'visual_index_lower_bound_bytes': minimum,
        'other_resident_costs_included': False,
        'status': 'INFEASIBLE' if rejected else 'UNDETERMINED',
        'quality_evaluated': False, 'accuracy': None,
        'reason': ('Visual indices alone exceed the complete-state budget' if rejected
                   else 'Passing this bound requires a full physical-layout audit'),
    }
