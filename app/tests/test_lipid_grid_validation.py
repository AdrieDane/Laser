"""The manual scoring graph separates class, ion form and RT isomers."""
import importlib.util
from pathlib import Path

import pandas as pd

MODULE=Path(__file__).resolve().parents[2]/'tools/validate_lipid_grid_evidence.py'
spec=importlib.util.spec_from_file_location('grid_validation',MODULE)
validation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_reference_graph_excludes_other_classes_adducts_and_distant_isomers():
    rows=[]
    for fid,formula,rt,lipid_class,adduct in [
        ('a','C10H20O2',300.,'PC','[M+H]+'),
        ('b','C11H22O2',309.6,'PC','[M+H]+'),
        ('c','C10H18O2',282.,'PC','[M+H]+'),
        ('isomer','C11H22O2',500.,'PC','[M+H]+'),
        ('other_class','C11H22O2',309.6,'PE','[M+H]+'),
        ('other_adduct','C11H22O2',309.6,'PC','[M+Na]+')]:
        rows.append(dict(stream='RP_Pos',feature_id=fid,formula=formula,manual_rt_seconds=rt,lipid_class=lipid_class,adduct=adduct))
    pairs=validation.reference_neighbours(pd.DataFrame(rows),.5)
    assert len(pairs)==6
    assert ('RP_Pos','b','c') in pairs
    assert not any('isomer' in pair or 'other_class' in pair or 'other_adduct' in pair for pair in pairs)
