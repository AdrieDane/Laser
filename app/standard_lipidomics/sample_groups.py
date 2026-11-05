"""Eligibility of study groups for biological significance comparisons."""
import re

_NON_STUDY = {'blank', 'blanco', 'qc', 'pool', 'pooled', 'is', 'standard',
              'calibrant', 'calibration', 'reference', 'qualitycontrol',
              'qcpool', 'processblank', 'solventblank'}
_STUDY_TYPES = {'study', 'sample', 'biological'}


def biological_label(label):
    value = str(label).casefold()
    tokens = set(re.findall(r'[a-z0-9]+', value))
    return not (tokens & _NON_STUDY or re.search(r'quality[ _-]*control', value))


def biological_group(features, group):
    """Reject QC/blank labels or types, including conflicting metadata."""
    if not biological_label(group):
        return False
    mapping = features.attrs.get('sample_group_map', {})
    types = features.attrs.get('sample_type_map', {})
    names = features.attrs.get('sample_group_name_map', {})
    samples = [sample for sample, value in mapping.items() if str(value) == str(group)]
    if any(not biological_label(names.get(sample, '')) for sample in samples):
        return False
    declared = {str(types.get(sample, '')).strip().casefold() for sample in samples}
    declared.discard('')
    return not declared or declared.issubset(_STUDY_TYPES)
