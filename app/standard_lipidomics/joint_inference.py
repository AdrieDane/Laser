"""Experimental MS1 hypotheses; heuristic costs, bounded families and abstention.

Reference lipid identities/adducts/charges never enter this module. Repeated
mass/isotope evidence is a single spectral block, not independent probabilities.
Homologous neighbours are interpretation context, never compound identity edges.
"""
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import asdict
import math

import numpy as np
import pandas as pd

from standard_lipidomics.acquisition_method import normalize_acquisition_method, chemistry_penalty
from standard_lipidomics.full_feature_evidence import _mass_pairs, profile_metrics, paired_columns
from standard_lipidomics.full_feature_groups import BoundedFamilies
from standard_lipidomics.isotope_quality import IsotopeIndex, score_envelope
from standard_lipidomics.isotope_patterns import parse_formula, formula_text, ion_composition
from standard_lipidomics.lipid_formulas import formula_candidates, atomic_masses, ELEMENTS, neutral_valid
from standard_lipidomics.msbuddy_runtime import freeze_runtime, run_native
from standard_lipidomics.untargeted_adducts import ADDUCTS, adduct_options

VERSION = 1
SETTINGS = {'native_weight': .25, 'chemistry_weight': .35, 'local_context_weight': .75,
            'cross_phase_context_weight': .40, 'margin': .25, 'abstain_cost': 3.,
            'target_candidates_per_ion': 2, 'component_root_limit': 32, 'beam_width': 64,
            'direct_options_per_root': 3, 'native_timeout_seconds': 1200}
IONS = {a.name: a for a in ADDUCTS}


def key(stream, fid):
    return str(stream), str(fid)


def label(candidate):
    return candidate['formula'], candidate['adduct']


def _signed(ion):
    return -ion.charge if ion.polarity == 'neg' else ion.charge


def _cost_number(value, fallback=8.):
    return float(value) if value is not None and np.isfinite(value) else fallback


def select_components(options, settings=SETTINGS):
    """Beam-search coupled assignments, including an explicit no-call choice.

    A feature cannot carry two formulas/adducts in any state. Oversize components
    remain unresolved; a lexicographic tie is never presented as confidence.
    """
    roots = sorted(options)
    parent = {r: r for r in roots}
    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r
    owners = {}
    for root in roots:
        for option in options[root][:settings['direct_options_per_root']]:
            for member in option['members']:
                if member in owners:
                    a, b = find(root), find(owners[member])
                    parent[max(a, b)] = min(a, b)
                else:
                    owners[member] = root
    groups = defaultdict(list)
    for root in roots:
        groups[find(root)].append(root)
    selected, audits = {}, []
    for number, component in enumerate(groups.values(), 1):
        ident = f'joint_component_{number:05d}'
        if len(component) > settings['component_root_limit']:
            for root in component:
                selected[root] = {'status': 'component budget exceeded', 'label': None, 'margin': None, 'component': ident}
            audits.append({'component': ident, 'roots': len(component), 'status': 'component budget exceeded', 'beam_pruned': False})
            continue
        # state: cost, physical-feature assignments, explicit root decisions
        states, pruned = [(0., {}, {})], False
        for root in component:
            expanded = {}
            for cost, assignments, decisions in states:
                forced = assignments.get(root)
                choices = ([o for o in options[root] if label(o) == forced] if forced else
                           [*options[root][:settings['direct_options_per_root']], None])
                for option in choices:
                    proposed = option['members'] if option is not None else {}
                    if any((m in assignments and assignments[m] != lab) or
                           (m in decisions and decisions[m] != lab) for m, lab in proposed.items()):
                        continue
                    decision = label(option) if option is not None else None
                    if root in assignments and assignments[root] != decision:
                        continue
                    mapping, chosen = {**assignments, **proposed}, {**decisions, root: decision}
                    value = cost + (option['joint_cost'] if option is not None else settings['abstain_cost'])
                    signature = (tuple(sorted(mapping.items())), tuple(sorted(chosen.items())))
                    if signature not in expanded or value < expanded[signature][0]:
                        expanded[signature] = (value, mapping, chosen)
            ordered = sorted(expanded.values(), key=lambda s: (s[0], repr(sorted(s[1].items())), repr(sorted(s[2].items()))))
            pruned |= len(ordered) > settings['beam_width']
            states = ordered[:settings['beam_width']]
        if not states:
            states = [(len(component)*settings['abstain_cost'], {}, dict.fromkeys(component))]
        best_cost, best, decisions = states[0]
        for member in set(best) | set(component):
            winner = decisions.get(member, best.get(member))
            alternatives = [cost-best_cost for cost, mapping, chosen in states[1:]
                            if chosen.get(member, mapping.get(member)) != winner]
            margin = min(alternatives) if alternatives else None
            resolved = winner is not None and ((margin is None and not pruned) or
                                               (margin is not None and margin >= settings['margin']))
            entry = {'status': 'tentative joint assignment' if resolved else 'abstained; competing or weak hypotheses',
                     'label': winner if resolved else None, 'margin': margin, 'component': ident}
            # A context ion can appear in separate root components. Conflicting
            # interpretations stay unresolved instead of letting order win.
            if member in selected and selected[member]['label'] != entry['label']:
                entry.update(status='unresolved context conflict', label=None)
            selected[member] = entry
        audits.append({'component': ident, 'roots': len(component), 'status': 'beam search complete',
                       'beam_pruned': pruned, 'retained_states': len(states), 'best_cost': best_cost})
    return selected, pd.DataFrame(audits)


class JointEngine:
    def __init__(self, result, analysis, criteria, progress=None):
        self.result, self.analysis, self.criteria, self.progress = result, analysis, criteria, progress
        self.method = normalize_acquisition_method(analysis.get('acquisition_method', result.get('acquisition_method')))
        self.chemistry = deepcopy(self.method)
        self.chemistry['adduct_scoring'] = 'method_aware'
        self.indexes = {s: IsotopeIndex(s, v['features']) for s, v in result['streams'].items()}
        self.queries = analysis['raw_p_features'].drop_duplicates(['stream', 'feature_id']).copy()
        self.queries['feature_id'] = self.queries.feature_id.astype(str)
        self.queries = self.queries.sort_values(['stream', 'feature_id'], kind='stable')
        self.root_keys = {key(r.stream, r.feature_id) for r in self.queries.itertuples()}
        self.ions = {s: adduct_options('neg' if s.endswith('Neg') else 'pos', method=self.method, stream=s) for s in self.indexes}
        self.mass = dict(zip(ELEMENTS, atomic_masses()))
        self.spectral, self.sources, self.support, self.budgets = {}, {}, [], []
        self.native_candidates, self.native_cases = [], []
        self.native = {}
        self.native_provenance = {'status': 'not requested'}

    def report(self, fraction, message):
        if self.progress:
            self.progress(fraction, message)

    def native_predictions(self):
        if not self.criteria.use_msbuddy:
            return
        cases, mapping = [], {}
        for number, row in enumerate(self.queries.itertuples()):
            index = self.indexes[row.stream]
            lo, hi = np.searchsorted(index.mz[index.order], [row.mz-.01, row.mz+6.01])
            matched = index.order[lo:hi]
            matched = matched[np.abs(index.rt[matched]-row.rt) <= .05]
            areas = np.nanmean(index.levels[matched], axis=1) if index.samples else np.full(len(matched), np.nan)
            spectrum = [[float(index.mz[i]), float(v)] for i, v in zip(matched, areas) if np.isfinite(v) and v > 0]
            for ion_number, ion in enumerate(self.ions[row.stream]):
                qid = f'case_{number:06d}_{ion_number:02d}'
                mapping[qid] = (row.stream, row.feature_id, ion.name)
                cases.append({'query_id': qid, 'stream': row.stream, 'feature_id': row.feature_id,
                              'mz': float(row.mz), 'rt': float(row.rt), 'adduct': ion.name,
                              'charge': _signed(ion), 'ms1': spectrum})
        runtime = self.result.get('_msbuddy_runtime')
        if runtime is None:
            runtime = freeze_runtime()
        records, self.native_provenance = run_native(cases, runtime, self.criteria.cross_mode_ppm_tolerance,
            lambda f, m: self.report(.02+.13*f, m))
        for qid, (stream, fid, adduct) in mapping.items():
            record = records.get(qid, {'status': 'unavailable native hypothesis', 'candidates': []})
            self.native[key(stream, fid), adduct] = record
            self.native_cases.append({'stream': stream, 'feature_id': fid, 'adduct': adduct,
                                      'native_status': record['status'], 'candidate_count': len(record['candidates'])})
            for candidate in record['candidates']:
                self.native_candidates.append({'stream': stream, 'feature_id': fid, **candidate})

    def spectral_score(self, stream, fid, candidate):
        cache_key = stream, str(fid), candidate['formula'], candidate['adduct']
        if cache_key not in self.spectral:
            # Context searches try other ions of the source neutral formula.
            # A water-loss ion cannot be constructed from an oxygen-free formula.
            try:
                ion_composition(candidate['formula'], candidate['adduct'])
            except ValueError as error:
                self.spectral[cache_key] = {'score': None, 'status': 'invalid ion composition',
                    'ion_composition_error': str(error), 'spectral_block_cost': 8.,
                    'native_rank': None, 'native_status': 'invalid ion composition'}
                return self.spectral[cache_key]
            measured, _ = score_envelope(self.indexes[stream], fid, candidate['formula'], candidate['adduct'],
                self.criteria, self.method['methods'][stream[:2]]['instrument'])
            native = self.native.get((key(stream, fid), candidate['adduct']), {})
            rank = next((c['rank'] for c in native.get('candidates', []) if formula_text(parse_formula(c['formula'])) == candidate['formula']), None)
            native_cost = min(3., math.log2(rank)) if rank else 3.
            spectral = _cost_number(measured['score'])
            if native.get('status') == 'complete':
                spectral = (1-SETTINGS['native_weight'])*spectral + SETTINGS['native_weight']*native_cost
            measured.update(spectral_block_cost=spectral, native_rank=rank, native_status=native.get('status', 'not available'))
            self.spectral[cache_key] = measured
        return self.spectral[cache_key]

    def source_hypotheses(self):
        pool = []
        for number, row in enumerate(self.queries.to_dict('records')):
            stream, fid = row['stream'], row['feature_id']
            generated, audit = formula_candidates(row['mz'], stream, self.criteria, self.method)
            unique = {label(c): c for c in generated}
            for ion in self.ions[stream]:
                native = self.native.get((key(stream, fid), ion.name), {})
                for c in native.get('candidates', []):
                    try:
                        atoms = parse_formula(c['formula'])
                        formula = formula_text(atoms)
                        ion_composition(formula, ion.name)
                    except ValueError:
                        continue
                    mass = sum(self.mass[e]*n for e, n in atoms.items() if e in self.mass)
                    target = ion.neutral(row['mz'])
                    error = (mass-target)/target*1e6 if target > 0 else math.inf
                    if not neutral_valid(atoms) or abs(error) > self.criteria.cross_mode_ppm_tolerance:
                        continue
                    k = formula, ion.name
                    if k in unique:
                        unique[k]['origins'] += '; msbuddy'
                    else:
                        unique[k] = {'formula': formula, 'adduct': ion.name, 'charge': _signed(ion),
                            'neutral_mass': mass, 'mass_error_ppm': error, 'origins': 'msbuddy',
                            'lipid_classes': '', 'structure_prior': 1.}
            generated = sorted(unique.values(), key=lambda c: (c['structure_prior']+abs(c['mass_error_ppm'])/self.criteria.cross_mode_ppm_tolerance, label(c)))
            # Reserve two slots per ion before filling by the shared prescore.
            order, seen = [], set()
            for depth in (0, 1):
                for ion in self.ions[stream]:
                    local = [c for c in generated if c['adduct'] == ion.name]
                    if len(local) > depth:
                        c = local[depth]
                        order.append(c)
                        seen.add(label(c))
            order.extend(c for c in generated if label(c) not in seen)
            scored = []
            for i, candidate in enumerate(order):
                item = {'stream': stream, 'feature_id': fid, **candidate, 'isotope_scored': False,
                        'joint_rank': None, 'joint_cost': None, 'status': 'not scored; explicit source budget'}
                if i < self.criteria.formula_scoring_limit:
                    item.update(self.spectral_score(stream, fid, candidate), isotope_scored=True)
                    classes = candidate['lipid_classes'].split('; ') if candidate['lipid_classes'] else ['']
                    chemistry = min(chemistry_penalty(self.chemistry, stream, item['adduct'], cls) for cls in classes)
                    item['chemistry_cost'] = chemistry
                    item['base_cost'] = item['spectral_block_cost']+candidate['structure_prior']+SETTINGS['chemistry_weight']*chemistry
                    if item['status'] not in {'parent mass incompatible', 'invalid ion composition'}:
                        scored.append(item)
                pool.append(item)
            self.sources[key(stream, fid)] = scored
            self.budgets.append({'stream': stream, 'feature_id': fid, **audit,
                'generated_union': len(generated), 'scored_candidates': len(scored),
                'source_budget_reached': len(generated) > self.criteria.formula_scoring_limit})
            if number % 100 == 0 or number+1 == len(self.queries):
                self.report(.15+.4*(number+1)/max(1, len(self.queries)), f'Joint formula candidates: {number+1:,}/{len(self.queries):,}')
        self.pool = pool

    def physical_links(self):
        frames = []
        for source, left in self.indexes.items():
            roots = copy(left)
            wanted = set(self.queries.loc[self.queries.stream.eq(source), 'feature_id'])
            roots.order = left.order[np.isin(left.ids[left.order], list(wanted))]
            for target, right in self.indexes.items():
                same_phase = source[:2] == target[:2]
                cap = self.criteria.same_mode_adduct_rt_tolerance_min if source == target else self.criteria.opposite_polarity_rt_tolerance_min if same_phase else None
                pairing = paired_columns(left, right)
                for si in self.ions[source]:
                    for ti in self.ions[target]:
                        if source == target and si.name == ti.name:
                            continue
                        if si.fragment and ti.fragment:
                            continue
                        for a, b in _mass_pairs(roots, right, si, ti, self.criteria.cross_mode_ppm_tolerance*2, cap):
                            n, rho, within, method = profile_metrics(left, right, a, b, pairing)
                            frames.append(pd.DataFrame({'source_stream': source, 'source_feature_id': left.ids[a],
                                'target_stream': target, 'target_feature_id': right.ids[b],
                                'source_adduct': si.name, 'target_adduct': ti.name, 'source_mz': left.mz[a],
                                'target_mz': right.mz[b], 'source_rt': left.rt[a], 'target_rt': right.rt[b],
                                'rt_compared': same_phase, 'paired_sample_count': n, 'abundance_spearman': rho,
                                'within_group_correlation': within, 'pairing_method': method,
                                'profile_supported': (n >= 6) & (rho >= .6) & (within >= .2)}))
        self.links = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.edges = {k: g.to_dict('records') for k, g in self.links.groupby(['source_stream', 'source_feature_id', 'source_adduct'], sort=False)} if not self.links.empty else {}

    def bounded(self, members):
        for s, f in members:
            a = self.indexes[s]
            ai = a.by_id[f]
            for t, g in members:
                if s[:2] != t[:2]:
                    continue
                b = self.indexes[t]
                bi = b.by_id[g]
                cap = self.criteria.same_mode_adduct_rt_tolerance_min if s == t else self.criteria.opposite_polarity_rt_tolerance_min
                width = (a.width[ai]+b.width[bi])/2
                allowed = min(cap, max(.005, width)) if np.isfinite(width) else cap
                if abs(a.rt[ai]-b.rt[bi]) > allowed+1e-10:
                    return False
        return True

    def rank_hypotheses(self):
        for number, (root, hypotheses) in enumerate(self.sources.items()):
            for hypothesis in hypotheses:
                source, fid = root
                matched = defaultdict(list)
                for edge in self.edges.get((source, fid, hypothesis['adduct']), []):
                    ion = IONS[edge['target_adduct']]
                    error = (edge['target_mz']-ion.mz(hypothesis['neutral_mass']))/ion.mz(hypothesis['neutral_mass'])*1e6
                    if abs(error) <= self.criteria.cross_mode_ppm_tolerance and edge['profile_supported']:
                        e = {**edge, 'formula': hypothesis['formula'], 'formula_mass_error_ppm': error}
                        # One correlated profile block; no group fold changes.
                        e['profile_cost'] = (2-edge['abundance_spearman']-edge['within_group_correlation'])/2
                        e['pre_cost'] = abs(error)/self.criteria.cross_mode_ppm_tolerance+e['profile_cost']
                        matched[(edge['target_stream'], edge['target_adduct'])].append(e)
                members, chosen, cross = {root: label(hypothesis)}, [], []
                for candidates in matched.values():
                    candidates.sort(key=lambda e: (e['pre_cost'], e['target_feature_id']))
                    evaluated = []
                    for edge in candidates[:SETTINGS['target_candidates_per_ion']]:
                        target = key(edge['target_stream'], edge['target_feature_id'])
                        score = self.spectral_score(*target, {'formula': hypothesis['formula'], 'adduct': edge['target_adduct']})
                        cost = edge['pre_cost']+.25*min(8., score['spectral_block_cost'])
                        entry = {**edge, 'source_formula': hypothesis['formula'], 'target_spectral_cost': score['spectral_block_cost'],
                                 'context_cost': cost, 'target_status': score['status'], 'context_selected': False,
                                 'target_budget_reached': len(candidates) > SETTINGS['target_candidates_per_ion']}
                        self.support.append(entry)
                        if score['status'] not in {'parent mass incompatible', 'invalid ion composition'}:
                            evaluated.append(entry)
                    evaluated.sort(key=lambda e: (e['context_cost'], e['target_feature_id']))
                    if not evaluated:
                        continue
                    best = evaluated[0]
                    ambiguous = (len(candidates) > SETTINGS['target_candidates_per_ion'] or
                                 len(evaluated) > 1 and evaluated[1]['context_cost']-best['context_cost'] < SETTINGS['margin'])
                    if ambiguous:
                        continue
                    target = key(best['target_stream'], best['target_feature_id'])
                    proposed = hypothesis['formula'], best['target_adduct']
                    # Query roots need their own scored evidence, even when
                    # discovered through another source's context.
                    if target in self.root_keys and not any(label(c) == proposed for c in self.sources[target]):
                        continue
                    candidate_members = {**members, target: proposed}
                    if not self.bounded(candidate_members):
                        continue
                    best['context_selected'] = True
                    if target[0][:2] == source[:2]:
                        members = candidate_members
                        chosen.append(best)
                    else:
                        cross.append(best)
                intact = {s for (s, _), (_, a) in members.items() if not IONS[a].fragment}
                members = {k: v for k, v in members.items() if not IONS[v[1]].fragment or k[0] in intact}
                hypothesis['members'] = members
                if root not in members:
                    hypothesis['joint_cost'] = math.inf
                    hypothesis['status'] = 'water loss without same-mode intact parent'
                    continue
                local_support = defaultdict(float)
                for e in chosen:
                    confidence = max(0., 1-e['profile_cost'])*math.exp(-.25*min(8., e['target_spectral_cost']))
                    if key(e['target_stream'], e['target_feature_id']) in self.root_keys:
                        confidence *= .5  # Reciprocal query evidence is not two independent observations.
                    local_support[e['target_stream']] = max(local_support[e['target_stream']], confidence)
                cross_support = max((max(0., 1-e['profile_cost'])*math.exp(-.25*min(8., e['target_spectral_cost'])) for e in cross), default=0.)
                hypothesis.update(local_context_support=float(np.mean(list(local_support.values()))) if local_support else 0.,
                                  cross_phase_context_support=cross_support,
                                  supporting_feature_count=len(members)-1)
                hypothesis['joint_cost'] = hypothesis['base_cost']-SETTINGS['local_context_weight']*hypothesis['local_context_support']-SETTINGS['cross_phase_context_weight']*cross_support
            ranked = sorted([h for h in hypotheses if np.isfinite(h['joint_cost'])], key=lambda h: (h['joint_cost'], label(h)))
            self.sources[root] = ranked
            for rank, h in enumerate(ranked, 1):
                h['joint_rank'] = rank
            if number % 100 == 0 or number+1 == len(self.sources):
                self.report(.6+.3*(number+1)/max(1, len(self.sources)), f'Joint evidence ranking: {number+1:,}/{len(self.sources):,}')

    def extend_cross_context(self, selected):
        conditional = defaultdict(set)
        for e in self.support:
            a, b = key(e['source_stream'], e['source_feature_id']), key(e['target_stream'], e['target_feature_id'])
            if (e['context_selected'] and a[0][:2] != b[0][:2] and b not in self.root_keys
                    and selected.get(a, {}).get('label') == (e['source_formula'], e['source_adduct'])):
                conditional[b].add((e['source_formula'], e['target_adduct']))
        for target, labels in conditional.items():
            if len(labels) == 1 and (target not in selected or selected[target]['label'] in labels):
                selected[target] = {'label': next(iter(labels)), 'status': 'tentative conditional cross-phase assignment',
                                    'component': 'cross-phase context', 'margin': None}
            else:
                selected[target] = {'label': None, 'status': 'conflicting cross-phase formulas/adducts',
                                    'component': 'cross-phase context', 'margin': None}

    def assemble(self, selected):
        assembler = BoundedFamilies(self.indexes, self.criteria)
        self.extend_cross_context(selected)
        selected_edges = []
        for e in self.support:
            a, b = key(e['source_stream'], e['source_feature_id']), key(e['target_stream'], e['target_feature_id'])
            if not e['context_selected'] or a[0][:2] != b[0][:2]:
                continue
            sa, sb = selected.get(a, {}), selected.get(b, {})
            if sa.get('label') != (e['source_formula'], e['source_adduct']) or sb.get('label') != (e['source_formula'], e['target_adduct']):
                continue
            selected_edges.append(e)
        def join(e):
            s, t = e['source_stream'], e['target_stream']
            a = assembler.offsets[s]+self.indexes[s].by_id[str(e['source_feature_id'])]
            b = assembler.offsets[t]+self.indexes[t].by_id[str(e['target_feature_id'])]
            si, ti = IONS[e['source_adduct']], IONS[e['target_adduct']]
            for node, stream, ion, other in ((a, s, si, ti), (b, t, ti, si)):
                if not ion.fragment:
                    continue
                nodes = assembler.members.get(assembler.root(node), {node})
                intact = any(assembler.mode[n] == stream and assembler.adduct[n] and not IONS[assembler.adduct[n]].fragment for n in nodes)
                if not intact and not (s == t and not other.fragment):
                    return 'water loss without same-mode intact parent'
            return assembler.join(a, b, si.name, ti.name, si.neutral(e['source_mz']), ti.neutral(e['target_mz']),
                                  max(0., 1-e['profile_cost']))
        # Non-significant opposite-phase ions still need an independently
        # checked local phase family. Never transfer an RP coelution interval
        # to NP, or union ions solely because a source proposed their formula.
        phase_members = defaultdict(list)
        for k, entry in selected.items():
            if entry.get('label'):
                phase_members[k[0][:2], entry['label'][0]].append(k)
        pairing_cache = {}
        for (_, formula), members in phase_members.items():
            for i, a in enumerate(sorted(members)):
                for b in sorted(members)[i+1:]:
                    sa, sb = selected[a]['label'][1], selected[b]['label'][1]
                    if (a[0] == b[0] and sa == sb) or not self.bounded({a: selected[a]['label'], b: selected[b]['label']}):
                        continue
                    left, right = self.indexes[a[0]], self.indexes[b[0]]
                    if (a[0], b[0]) not in pairing_cache:
                        pairing_cache[a[0], b[0]] = paired_columns(left, right)
                    ai, bi = left.by_id[a[1]], right.by_id[b[1]]
                    n, rho, within, _ = profile_metrics(left, right, np.array([ai]), np.array([bi]), pairing_cache[a[0], b[0]])
                    if n[0] < 6 or rho[0] < .6 or within[0] < .2 or not np.isfinite(rho[0]+within[0]):
                        continue
                    if IONS[sa].fragment and IONS[sb].fragment:
                        continue
                    e = {'source_stream': a[0], 'source_feature_id': a[1], 'source_adduct': sa,
                         'target_stream': b[0], 'target_feature_id': b[1], 'target_adduct': sb,
                         'source_mz': left.mz[ai], 'target_mz': right.mz[bi],
                         'source_rt': left.rt[ai], 'target_rt': right.rt[bi], 'rt_compared': True,
                         'source_formula': formula, 'profile_cost': float((2-rho[0]-within[0])/2),
                         'context_cost': 0., 'context_selected': True, 'conditional_phase_family': True}
                    selected_edges.append(e)
        for e in sorted(selected_edges, key=lambda e: (int(IONS[e['source_adduct']].fragment or IONS[e['target_adduct']].fragment),
                e['source_stream'] != e['target_stream'], e['context_cost'], e['source_stream'], e['source_feature_id'], e['target_feature_id'])):
            e['joint_link_status'] = join(e)
        for k, entry in selected.items():
            if entry.get('label') and IONS[entry['label'][1]].fragment:
                node = assembler.offsets[k[0]]+self.indexes[k[0]].by_id[k[1]]
                nodes = assembler.members.get(assembler.root(node), {node})
                if not any(assembler.mode[n] == k[0] and assembler.adduct[n] and not IONS[assembler.adduct[n]].fragment for n in nodes):
                    entry.update(label=None, status='water loss without same-mode intact parent')
        # Cross-phase edges remain one-to-many proposals until both phase
        # families have a unique reciprocal partner. Formula equality alone
        # supplies no link, and no RP-to-NP RT comparison is invented.
        phase_ledger = assembler.table()
        family = {key(r.stream, r.feature_id): r.compound_id for r in phase_ledger.itertuples()}
        cross_pairs = {}
        for e in self.support:
            a, b = key(e['source_stream'], e['source_feature_id']), key(e['target_stream'], e['target_feature_id'])
            if a[0][:2] == b[0][:2]:
                continue
            if selected.get(a, {}).get('label') != (e['source_formula'], e['source_adduct']) or selected.get(b, {}).get('label') != (e['source_formula'], e['target_adduct']):
                continue
            pair = tuple(sorted((family[a], family[b])))
            if pair not in cross_pairs or e['context_cost'] < cross_pairs[pair]['context_cost']:
                cross_pairs[pair] = e
        alternatives = defaultdict(list)
        for pair, e in cross_pairs.items():
            for f in pair:
                alternatives[f].append((e['context_cost'], pair))
        unique = {}
        for f, choices in alternatives.items():
            choices.sort()
            if len(choices) == 1 or choices[1][0]-choices[0][0] >= SETTINGS['margin']:
                unique[f] = choices[0][1]
        associations = []
        for pair, e in sorted(cross_pairs.items()):
            status = 'ambiguous cross-phase association; retained separately'
            if all(unique.get(f) == pair for f in pair):
                status = join(e)
            associations.append({**e, 'source_phase_family': family[key(e['source_stream'], e['source_feature_id'])],
                                 'target_phase_family': family[key(e['target_stream'], e['target_feature_id'])],
                                 'association_status': status})
        ledger = assembler.table()
        assignments = []
        for row in ledger.itertuples():
            k = key(row.stream, row.feature_id)
            entry = selected.get(k, {'status': 'not queried or unresolved', 'label': None, 'margin': None, 'component': ''})
            formula, adduct = entry['label'] or ('', '')
            assignments.append({'stream': row.stream, 'feature_id': row.feature_id, 'joint_formula': formula,
                'joint_adduct': adduct, 'joint_status': entry['status'], 'joint_margin': entry['margin'],
                'joint_component': entry['component'], 'joint_charge': _signed(IONS[adduct]) if adduct else None})
        ledger = ledger.merge(pd.DataFrame(assignments), on=['stream', 'feature_id'], validate='one_to_one')
        # A resolved singleton still carries a formula/adduct hypothesis.
        ledger['adduct'] = ledger.joint_adduct
        ledger['predicted_chemical_formula'] = ledger.joint_formula
        ledger['formula_status'] = ledger.joint_status
        ledger['association_status'] = ledger.joint_status
        hits = self.root_keys
        chosen_groups = set(ledger.loc[[key(r.stream, r.feature_id) in hits for r in ledger.itertuples()], 'compound_id'])
        compounds = ledger[ledger.compound_id.isin(chosen_groups)].copy()
        compounds['raw_p_significant'] = [key(r.stream, r.feature_id) in hits for r in compounds.itertuples()]
        stats = self.analysis['features'][['stream', 'feature_id', 'p_value', 'log2_fold_change_b_vs_a']].copy()
        stats['feature_id'] = stats.feature_id.astype(str)
        compounds = compounds.merge(stats, on=['stream', 'feature_id'], how='left', validate='one_to_one')
        return ledger, compounds, pd.DataFrame(associations), pd.DataFrame(selected_edges)


def apply_joint_inference(result, analysis, criteria, progress=None):
    if criteria.joint_model == 'current':
        return analysis
    if analysis.get('raw_p_features', pd.DataFrame()).empty:
        return {**analysis, 'joint_provenance': {'version': VERSION, 'status': 'no queries', 'default_changed': False},
                'msbuddy_provenance': {'status': 'no queries'}}
    engine = JointEngine(result, analysis, criteria, progress)
    engine.report(.01, 'Preparing blind formula/adduct hypotheses')
    engine.native_predictions()
    engine.source_hypotheses()
    engine.report(.56, 'Pairing biological profiles for formula-compatible ions')
    engine.physical_links()
    engine.rank_hypotheses()
    selected, components = select_components(engine.sources)
    ledger, compounds, associations, links = engine.assemble(selected)
    # Members are structured solver objects, exported in normalized support
    # tables; never stringify executable objects into project CSVs.
    hypotheses = pd.DataFrame([{k: v for k, v in h.items() if k != 'members'} for h in engine.pool])
    return {**analysis, 'joint_hypotheses': hypotheses, 'joint_feature_ledger': ledger,
            'joint_compound_features': compounds, 'joint_relationships': engine.links,
            'joint_support': pd.DataFrame(engine.support), 'joint_associations': associations,
            'joint_selected_links': links, 'joint_component_audit': components,
            'joint_search_audit': pd.DataFrame(engine.budgets),
            'msbuddy_candidates': pd.DataFrame(engine.native_candidates), 'msbuddy_cases': pd.DataFrame(engine.native_cases),
            'joint_provenance': {'version': VERSION, 'settings': dict(SETTINGS), 'criteria': asdict(criteria),
                'query_scope': 'all requested raw-p features; physical support includes all original features',
                'reference_identities_used': False, 'reference_adducts_or_charges_used': False,
                'mass_basis': 'supplied feature coordinates; tolerance-normalized errors, no fitted calibration uncertainty',
                'correlation': 'biological paired profiles with group means removed; not chromatographic shape',
                'spectral_block': 'correlated complete-isotope/native evidence averaged, not multiplied as probabilities',
                'confidence': 'heuristic cost margins over retained beam states; not calibrated probabilities',
                'homologues': 'legacy local grids retained as context only; never identity unions',
                'default_changed': False}, 'msbuddy_provenance': engine.native_provenance}
