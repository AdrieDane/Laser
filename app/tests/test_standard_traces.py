import base64
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from standard_traces import (
    _extract_target_traces,
    _iter_ms1_scans,
    _parse_retention_time,
    discover_xcms_raw_files,
)


def _encoded(values, *, endian="<"):
    raw = struct.pack(endian + f"{len(values)}f", *values)
    return base64.b64encode(raw).decode()


class StandardTraceReaderTests(unittest.TestCase):
    def test_retention_time_parser_handles_mzxml_duration_and_seconds(self):
        self.assertAlmostEqual(_parse_retention_time("PT90S"), 1.5)
        self.assertAlmostEqual(_parse_retention_time("120"), 2.0)

    def test_mzxml_reads_ms1_only_and_extracts_centroid_eic(self):
        xml = f"""<?xml version='1.0'?>
<mzXML><msRun>
  <scan num='1' msLevel='1' retentionTime='PT60S'>
    <peaks precision='32' byteOrder='network' pairOrder='m/z-int'
           compressionType='none'>{_encoded([100.0, 2.0, 200.0, 5.0], endian='>')}</peaks>
  </scan>
  <scan num='2' msLevel='2' retentionTime='PT61S'>
    <peaks precision='32' byteOrder='network' pairOrder='m/z-int'
           compressionType='none'>{_encoded([100.0, 99.0], endian='>')}</peaks>
  </scan>
  <scan num='3' msLevel='1' retentionTime='PT62S'>
    <peaks precision='32' byteOrder='network' pairOrder='m/z-int'
           compressionType='none'>{_encoded([100.0, 4.0], endian='>')}</peaks>
  </scan>
</msRun></mzXML>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mzXML"
            path.write_text(xml)
            scans = list(_iter_ms1_scans(path))
            self.assertEqual(len(scans), 2)
            traces = _extract_target_traces(path, (100.0,), 5.0)
            np.testing.assert_allclose(traces[0][:, 0], [1.0, 1.0333333333])
            np.testing.assert_allclose(traces[0][:, 1], [2.0, 4.0])

    def test_mzml_reads_minutes_and_binary_arrays(self):
        mz = _encoded([100.0, 200.0])
        intensity = _encoded([3.0, 7.0])
        xml = f"""<?xml version='1.0'?>
<mzML><run><spectrumList count='1'><spectrum id='scan=1'>
  <cvParam name='ms level' value='1'/>
  <scanList><scan><cvParam name='scan start time' value='2.5' unitName='minute'/></scan></scanList>
  <binaryDataArrayList>
    <binaryDataArray><cvParam name='m/z array'/><cvParam name='32-bit float'/><binary>{mz}</binary></binaryDataArray>
    <binaryDataArray><cvParam name='intensity array'/><cvParam name='32-bit float'/><binary>{intensity}</binary></binaryDataArray>
  </binaryDataArrayList>
</spectrum></spectrumList></run></mzML>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mzML"
            path.write_text(xml)
            scans = list(_iter_ms1_scans(path))
            self.assertEqual(len(scans), 1)
            self.assertAlmostEqual(scans[0][0], 2.5)
            np.testing.assert_allclose(scans[0][1], [100.0, 200.0])
            np.testing.assert_allclose(scans[0][2], [3.0, 7.0])

    def test_raw_file_discovery_maps_case_insensitive_stream_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "Data" / "Processed" / "RP_Pos" / "Group"
            raw.mkdir(parents=True)
            (raw / "Sample_A.mzXML").write_text("<mzXML/>")
            discovered = discover_xcms_raw_files(root / "Processing" / "Annotation", "RP_POS")
            self.assertEqual(set(discovered), {"sample_a"})


if __name__ == "__main__":
    unittest.main()
