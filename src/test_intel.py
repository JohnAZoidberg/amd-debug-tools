#!/usr/bin/python3
# SPDX-License-Identifier: MIT

"""
This module contains unit tests for the Intel helper functions in the amd-debug-tools package.
"""

import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from amd_debug.intel import (
    IntelPmcCore,
    blocker_method,
    diff_counters,
    substate_blocker_counts,
    find_substate_blockers,
    load_cycle_state,
    lpit_supported,
    parse_package_cstates,
    parse_substate_requirements,
    parse_substate_residencies,
    parse_substate_status,
    read_lpit_residency,
    requirement_counters,
    save_cycle_state,
    substate_order,
)

REQUIREMENTS = """PMC0
                       Element |    S0i2.0 |    S0i2.1 |    S0i2.2 |    Status |
                      PMC_LPM0 |  Required |  Required |  Required |       Yes |
                       XHCI_D3 |  Required |  Required |  Required |           |
                       SATA_D3 |           |  Required |  Required |       Yes |
                       PCIE_L1 |           |           |  Required |           |
"""


COUNTER_REQUIREMENTS = """PMC0
                                 Element |    S0i2.0 |    S0i2.1 |     Value |
pmc0:                    XHCI_PGD0_PG_STS |           |  Required |       312 |
pmc0:                   CLINK_PGD0_PG_STS |  Required |  Required |  64403613 |
pmc0:                     CSE_PGD0_PG_STS |  Required |  Required |   1544512 |
pmc0:                              Memory |  Required |  Required |        16 |
"""

STATUS_REGISTERS = """
PMC0:LPM_STATUS_0:\t0x333c1144
PMC0:AON2_OFF_STS                0
PMC0:CLINK_PGD0_PG_STS           0
PMC0:CSE_PGD0_PG_STS             1
"""


class TestIntelParsers(unittest.TestCase):
    """Test the pmc_core file parsers"""

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(filename="/dev/null", level=logging.DEBUG)

    def test_parse_substate_residencies_plain(self):
        """Test parsing the plain substate residency layout"""
        text = "Substate   Residency\nS0i2.0     123\nS0i2.1     0\nS0i3.0     42\n"
        self.assertEqual(
            parse_substate_residencies(text),
            {"S0i2.0": 123, "S0i2.1": 0, "S0i3.0": 42},
        )

    def test_parse_substate_residencies_enabled(self):
        """Test parsing the layout with an Enabled column"""
        text = "Substate Residency\nEnabled S0i2.0 5\nDisabled S0i2.1 7\n"
        self.assertEqual(parse_substate_residencies(text), {"S0i2.0": 5, "S0i2.1": 7})

    def test_parse_substate_residencies_garbage(self):
        """Test parsing garbage"""
        self.assertEqual(parse_substate_residencies(""), {})
        self.assertEqual(parse_substate_residencies("foo bar\n\n"), {})

    def test_parse_package_cstates(self):
        """Test parsing package_cstate_show"""
        text = "Package C2 : 100\nPackage C10 : 200\nbogus\nPackage C3 : nope\n"
        self.assertEqual(
            parse_package_cstates(text), {"Package C2": 100, "Package C10": 200}
        )

    def test_parse_substate_requirements(self):
        """Test parsing substate_requirements"""
        parsed = parse_substate_requirements(REQUIREMENTS)
        self.assertEqual(parsed["modes"], ["S0i2.0", "S0i2.1", "S0i2.2"])
        self.assertEqual(len(parsed["rows"]), 4)
        self.assertEqual(parsed["rows"][0]["element"], "PMC_LPM0")
        self.assertEqual(parsed["rows"][0]["status"], "Yes")
        self.assertEqual(parsed["rows"][1]["required"], {"S0i2.0", "S0i2.1", "S0i2.2"})
        self.assertEqual(parsed["rows"][2]["required"], {"S0i2.1", "S0i2.2"})

    def test_parse_substate_requirements_without_header(self):
        """Rows before a header are ignored"""
        parsed = parse_substate_requirements("| foo | Required | Yes |\n")
        self.assertEqual(parsed, {"modes": [], "format": "", "rows": []})

    def test_find_substate_blockers(self):
        """Test finding blockers for a given substate"""
        parsed = parse_substate_requirements(REQUIREMENTS)
        self.assertEqual(find_substate_blockers(parsed, "S0i2.0"), ["XHCI_D3"])
        self.assertEqual(find_substate_blockers(parsed, "S0i2.1"), ["XHCI_D3"])
        self.assertEqual(
            find_substate_blockers(parsed, "S0i2.2"), ["XHCI_D3", "PCIE_L1"]
        )
        self.assertEqual(find_substate_blockers({}, "S0i2.0"), [])

    def test_parse_substate_requirements_counters(self):
        """Kernels with S0ix blocker counters use a Value column"""
        parsed = parse_substate_requirements(COUNTER_REQUIREMENTS)
        self.assertEqual(parsed["format"], "counter")
        self.assertEqual(parsed["modes"], ["S0i2.0", "S0i2.1"])
        self.assertEqual(parsed["rows"][1]["element"], "pmc0: CLINK_PGD0_PG_STS")
        self.assertEqual(parsed["rows"][1]["counter"], 64403613)
        self.assertEqual(
            requirement_counters(parsed),
            {
                "pmc0: XHCI_PGD0_PG_STS": 312,
                "pmc0: CLINK_PGD0_PG_STS": 64403613,
                "pmc0: CSE_PGD0_PG_STS": 1544512,
                "pmc0: Memory": 16,
            },
        )
        self.assertEqual(parse_substate_requirements(REQUIREMENTS)["format"], "status")

    def test_find_substate_blockers_counters(self):
        """Counter tables report the IPs whose counter advanced"""
        parsed = parse_substate_requirements(COUNTER_REQUIREMENTS)
        before = {
            "pmc0: XHCI_PGD0_PG_STS": 300,
            "pmc0: CLINK_PGD0_PG_STS": 64400000,
            "pmc0: CSE_PGD0_PG_STS": 1544512,
            "pmc0: Memory": 16,
        }
        self.assertEqual(
            find_substate_blockers(parsed, "S0i2.0", before=before),
            ["pmc0: CLINK_PGD0_PG_STS"],
        )
        # XHCI is only required for S0i2.1 and its counter advanced too
        self.assertEqual(
            find_substate_blockers(parsed, "S0i2.1", before=before),
            ["pmc0: XHCI_PGD0_PG_STS", "pmc0: CLINK_PGD0_PG_STS"],
        )
        self.assertEqual(
            substate_blocker_counts(parsed, "S0i2.1", before),
            {"pmc0: XHCI_PGD0_PG_STS": 12, "pmc0: CLINK_PGD0_PG_STS": 3613},
        )
        self.assertEqual(substate_blocker_counts(parsed, "S0i2.1", None), {})
        # a cumulative counter alone says nothing about this cycle
        self.assertEqual(find_substate_blockers(parsed, "S0i2.0"), [])
        self.assertEqual(blocker_method(parsed), "")
        self.assertEqual(
            blocker_method(parsed, before=before),
            "blocker counter advanced during the cycle",
        )

    def test_find_substate_blockers_latched_fallback(self):
        """Without a counter baseline the latched status bits are used"""
        parsed = parse_substate_requirements(COUNTER_REQUIREMENTS)
        status = parse_substate_status(STATUS_REGISTERS)
        self.assertEqual(status["CLINK_PGD0_PG_STS"], 0)
        self.assertEqual(status["pmc0: CSE_PGD0_PG_STS"], 1)
        self.assertNotIn("LPM_STATUS_0:", status)
        # Memory has no status bit so it is not reported either way
        self.assertEqual(
            find_substate_blockers(parsed, "S0i2.0", status=status),
            ["pmc0: CLINK_PGD0_PG_STS"],
        )
        self.assertEqual(blocker_method(parsed, status=status), "status latched")

    def test_substate_order(self):
        """Test ordering substates from shallowest to deepest"""
        self.assertEqual(
            substate_order(["S0i3.0", "S0i2.2", "S0i2.0", "S0i2.1"]),
            ["S0i2.0", "S0i2.1", "S0i2.2", "S0i3.0"],
        )

    def test_diff_counters(self):
        """Test differencing counter snapshots"""
        self.assertEqual(
            diff_counters({"a": 1, "b": 5}, {"a": 4, "b": 5, "c": 9}),
            {"a": 3, "b": 0},
        )
        self.assertEqual(diff_counters({}, {"a": 1}), {})


class TestLpit(unittest.TestCase):
    """Test the LPIT residency helpers"""

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(filename="/dev/null", level=logging.DEBUG)

    @patch("amd_debug.intel.read_file")
    def test_read_lpit_residency(self, mock_read_file):
        """Test reading both counters"""
        mock_read_file.side_effect = lambda p: (
            "1000" if p.endswith("cpu_residency_us") else "250"
        )
        self.assertEqual(read_lpit_residency(), {"cpu": 1000, "system": 250})

    @patch("amd_debug.intel.read_file", side_effect=FileNotFoundError)
    def test_read_lpit_residency_missing(self, _mock_read_file):
        """Test reading when LPIT isn't available"""
        self.assertEqual(read_lpit_residency(), {})

    @patch("amd_debug.intel.read_file", return_value="garbage")
    def test_read_lpit_residency_garbage(self, _mock_read_file):
        """Test reading garbage"""
        self.assertEqual(read_lpit_residency(), {})

    @patch("os.path.exists", return_value=True)
    def test_lpit_supported(self, mock_exists):
        """Test LPIT support detection"""
        self.assertTrue(lpit_supported())
        mock_exists.return_value = False
        self.assertFalse(lpit_supported())


class TestIntelPmcCore(unittest.TestCase):
    """Test the pmc_core debugfs wrapper using a temporary directory"""

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(filename="/dev/null", level=logging.DEBUG)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.debugfs = os.path.join(self.tmp.name, "pmc_core")
        self.params = os.path.join(self.tmp.name, "parameters")
        os.makedirs(self.debugfs)
        os.makedirs(self.params)
        self.pmc = IntelPmcCore(debugfs=self.debugfs, params=self.params)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, directory, name, content):
        with open(os.path.join(directory, name), "w", encoding="utf-8") as w:
            w.write(content)

    def test_available(self):
        """Test debugfs availability detection"""
        self.assertTrue(self.pmc.available)
        self.assertFalse(IntelPmcCore(debugfs="/nonexistent").available)

    def test_read_missing(self):
        """Test reading a missing file"""
        self.assertIsNone(self.pmc.read("substate_residencies"))
        self.assertEqual(self.pmc.substate_residencies(), {})
        self.assertEqual(self.pmc.package_cstates(), {})
        self.assertEqual(self.pmc.substate_requirements(), {})

    def test_read_permission_error(self):
        """Test reading a file that can't be read"""
        with patch("amd_debug.intel.read_file", side_effect=PermissionError):
            self.assertIsNone(self.pmc.read("substate_residencies"))
        with patch("amd_debug.intel.read_file", side_effect=OSError):
            self.assertIsNone(self.pmc.read("substate_residencies"))

    def test_parsed_reads(self):
        """Test the parsed accessors"""
        self._write(self.debugfs, "substate_residencies", "Substate Residency\nS0i2.0 5\n")
        self._write(self.debugfs, "package_cstate_show", "Package C10 : 7\n")
        self._write(self.debugfs, "substate_requirements", REQUIREMENTS)
        self.assertEqual(self.pmc.substate_residencies(), {"S0i2.0": 5})
        self.assertEqual(self.pmc.package_cstates(), {"Package C10": 7})
        self.assertEqual(
            self.pmc.substate_requirements()["modes"], ["S0i2.0", "S0i2.1", "S0i2.2"]
        )

    def test_write(self):
        """Test writing debugfs files"""
        self._write(self.debugfs, "lpm_latch_mode", "")
        self.assertTrue(self.pmc.write("lpm_latch_mode", "c10"))
        self.assertEqual(self.pmc.read("lpm_latch_mode"), "c10")
        self.assertFalse(self.pmc.write("missing/dir/file", "x"))

    def test_latch_mode(self):
        """Test latching clears then sets the mode"""
        self._write(self.debugfs, "lpm_latch_mode", "")
        with patch.object(self.pmc, "write", return_value=True) as mock_write:
            self.assertTrue(self.pmc.latch_mode("S0i2.0"))
            mock_write.assert_any_call("lpm_latch_mode", "clear")
            mock_write.assert_any_call("lpm_latch_mode", "S0i2.0")
        with patch.object(self.pmc, "write", return_value=False):
            self.assertFalse(self.pmc.latch_mode("c10"))

    def test_params(self):
        """Test module parameter access"""
        self.assertIsNone(self.pmc.get_param("warn_on_s0ix_failures"))
        self.assertFalse(self.pmc.set_param("missing/dir/param", "Y"))
        self._write(self.params, "warn_on_s0ix_failures", "N")
        self.assertEqual(self.pmc.get_param("warn_on_s0ix_failures"), "N")
        self.assertTrue(self.pmc.set_param("warn_on_s0ix_failures", "Y"))
        self.assertEqual(self.pmc.get_param("warn_on_s0ix_failures"), "Y")


class TestCycleState(unittest.TestCase):
    """Test persisting the pre-suspend snapshot for the systemd hook path"""

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(filename="/dev/null", level=logging.DEBUG)

    def test_round_trip(self):
        """Test save and load"""
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "state.json")
            state = {"lpit": {"cpu": 1, "system": 2}, "latch": "c10", "warn_orig": "N"}
            self.assertTrue(save_cycle_state(state, fname))
            self.assertEqual(load_cycle_state(fname), state)
            # the state is consumed on load
            self.assertFalse(os.path.exists(fname))
            self.assertEqual(load_cycle_state(fname), {})

    def test_bad_contents(self):
        """Test loading invalid contents"""
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "state.json")
            with open(fname, "w", encoding="utf-8") as w:
                w.write("[1, 2]")
            self.assertEqual(load_cycle_state(fname), {})
            with open(fname, "w", encoding="utf-8") as w:
                w.write("not json")
            self.assertEqual(load_cycle_state(fname), {})

    def test_save_failure(self):
        """Test saving to an unwritable location"""
        self.assertFalse(save_cycle_state({}, "/nonexistent/dir/state.json"))
