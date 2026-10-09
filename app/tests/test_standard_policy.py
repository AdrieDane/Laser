import tempfile
import unittest
from pathlib import Path

import pandas as pd

from annotator.reference import load_reference
from standard_lipidomics.workflow import annotate_standard_batch
from standard_lipidomics.policy import (
    apply_matlab_lipid_policy,
    discover_matlab_fa_ranges,
    discover_matlab_policy_file,
    load_matlab_lipid_policy,
)


POLICY_HEADER = (
    "class\tclass_name\tR_group\tcomposition\t\t\tpriority_NP\t"
    "priority_RP\tcombined\tmeasurement\tchem\n"
)


def write_policy(path: Path) -> Path:
    path.write_text(
        POLICY_HEADER
        + "TG\ttriacylglycerol\tFA\t\t\t\t\t10\t\t"
        "RP_pos_NH3; RP_pos_Na\tC6H5O6R3\n"
        + "PC\tphosphatidylcholine\tFA\t\t\t\t10\t10\t\t"
        "NP_pos_0; RP_pos_0\tC10H18NO8PR2\n"
        + "\t\t\t\t\tO+P-\t10\t-1\t\t"
        "NP_pos_0; RP_pos_0\tC10H20NO7PR2\n"
        + "\tether\t\t\t\tO-\t-1\t10\tPC_O\t"
        "NP_pos_0; RP_pos_0\tC10H20NO7PR2\n"
        + "\tplasmalogen\t\t\t\tP-\t-1\t10\tPC_O\t"
        "NP_pos_0; RP_pos_0\tC10H18NO7PR2\n"
        + "LPC\tlysophosphatidylcholine\tFA\t\t\t\t10\t-1\t\t"
        "NP_pos_0; RP_pos_0\tC9H19NO7PR\n"
        + "\t1-acyl\t\t1-acyl \t\t\t-1\t10\tLPC\t"
        "NP_pos_0; RP_pos_0\tC9H19NO7PR\n"
        + "DGDG\tdiglycosyl diacylglycerol\tFA\t\t\t\t\t\t\t"
        "RP_pos_NH3\tC17H26O15R2\n",
        encoding="latin1",
    )
    return path


def write_reference(path: Path) -> pd.DataFrame:
    pd.DataFrame({
        "name": [
            "TG(30:0)", "TG(130:0)", "PC(O-34:1)",
            "1-acyl LPC(16:0)", "DGDG(34:1)", "cholesterol",
        ],
        "class": ["TG", "TG", "PC", "LPC", "DGDG", "cholesterol"],
        "formula": [""] * 6,
        "adduct": [
            "[M+NH4]+", "[M+NH4]+", "[M+H]+", "[M+H]+", "[M+NH4]+",
            "[M+NH4]+",
        ],
        "charge": [1] * 6,
        "mz": [500.0, 501.0, 760.0, 496.0, 800.0, 369.0],
        "RT_RP_median_min": [8.0] * 6,
        "n_RT": [10] * 6,
    }).to_csv(path, index=False)
    return load_reference(path)


class StandardPolicyTests(unittest.TestCase):
    def test_policy_enforces_priority_composition_and_phase_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = load_matlab_lipid_policy(
                write_policy(root / "lipids.tsv"),
                fa_carbon_range=(8, 42),
                fa_db_range=(0, 8),
            )
            reference = write_reference(root / "reference.csv")

            rp, rp_audit = apply_matlab_lipid_policy(
                reference, "RP_Pos", policy
            )
            self.assertEqual(
                set(rp["name"]),
                {"TG(30:0)", "PC(O-34:1)", "1-acyl LPC(16:0)"},
            )
            self.assertEqual(
                rp.set_index("name").loc["PC(O-34:1)", "matlab_policy_class"],
                "PC[O]",
            )
            self.assertEqual(
                rp_audit.set_index("name").loc[
                    "TG(130:0)", "policy_result"
                ],
                "composition_out_of_range",
            )
            self.assertEqual(
                rp_audit.set_index("name").loc[
                    "DGDG(34:1)", "policy_result"
                ],
                "nonpositive_phase_priority",
            )

            np_reference, _ = apply_matlab_lipid_policy(
                reference, "NP_Pos", policy
            )
            selected = np_reference.set_index("name")
            self.assertEqual(
                selected.loc["PC(O-34:1)", "matlab_policy_class"],
                "PC[O+P]",
            )
            self.assertEqual(
                selected.loc[
                    "1-acyl LPC(16:0)", "matlab_policy_class"
                ],
                "LPC",
            )

    def test_project_discovery_finds_policy_and_fa_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "Processing" / "Annotation"
            database = root / "Data" / "Meta" / "MS_database"
            annotation.mkdir(parents=True)
            database.mkdir(parents=True)
            policy_path = write_policy(database / "lipids.tsv")
            script = annotation / "analyse_MS_data.m"
            script.write_text(
                "e=MS_analysis('set_param','R_group.FA.range',[10,40;0,6]);\n"
            )

            self.assertEqual(
                discover_matlab_policy_file(annotation), policy_path
            )
            carbon, db, source = discover_matlab_fa_ranges(annotation)
            self.assertEqual(carbon, (10, 40))
            self.assertEqual(db, (0, 6))
            self.assertEqual(source, script)

    def test_batch_exports_selected_policy_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = load_matlab_lipid_policy(
                write_policy(root / "lipids.tsv")
            )
            reference = write_reference(root / "reference.csv")
            source = (
                b"name\tmzmed\trtmed\tnpeaks\tGroup_01_wt\tsample_a\n"
                b"1\t500.0\t8.0\t10\t1\t100\n"
            )
            result = annotate_standard_batch(
                {"RP_Pos": source},
                reference,
                adapt_class_rt=False,
                matlab_policy=policy,
            )

            candidate = result["all_candidates"].iloc[0]
            self.assertTrue(bool(candidate["matlab_policy_applied"]))
            self.assertEqual(candidate["matlab_policy_class"], "TG")
            self.assertEqual(
                candidate["matlab_policy_measurement"], "RP_pos_NH3"
            )
            self.assertFalse(result["policy_audit"].empty)


if __name__ == "__main__":
    unittest.main()
