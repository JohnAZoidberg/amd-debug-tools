#!/usr/bin/python3
# SPDX-License-Identifier: MIT

"""
This module contains unit tests for the s2idle tool in the amd-debug-tools package.
"""

import math
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
import numpy as np
import pandas as pd
from markupsafe import Markup

from amd_debug.sleep_report import (
    remove_duplicates,
    format_gpio_as_str,
    format_irq_as_str,
    format_as_human,
    format_as_seconds,
    format_watts,
    format_percent,
    format_timedelta,
    parse_hw_sleep,
    get_energy_window,
    SleepReport,
)

from amd_debug.wake import WakeGPIO, WakeIRQ


class TestSleepReportUtils(unittest.TestCase):
    """Unit tests for the sleep report utilities."""

    def test_remove_duplicates(self):
        """Test the remove_duplicates function."""
        self.assertEqual(remove_duplicates("1, 2, 2, 3"), [1, 2, 3])
        self.assertEqual(remove_duplicates("4 4 5 6"), [4, 5, 6])
        self.assertEqual(remove_duplicates(""), [])

    def test_format_gpio_as_str(self):
        """Test the format_gpio_as_str function."""
        self.assertEqual(format_gpio_as_str("1, 2, 2, 3"), "1, 2, 3")
        self.assertEqual(format_gpio_as_str("4 4 5 6"), "4, 5, 6")
        self.assertEqual(format_gpio_as_str(""), "")

    @patch("amd_debug.wake.read_file")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.walk")
    def test_format_irq_as_str(
        self, _mock_os_walk, _mock_os_listdir, mock_os_path_exists, mock_read_file
    ):
        """Test the format_irq_as_str function."""
        mock_read_file.side_effect = lambda path: {
            "/sys/kernel/irq/20/chip_name": "",
            "/sys/kernel/irq/20/actions": "",
            "/sys/kernel/irq/20/wakeup": "disabled",
        }.get(path, "")

        # Mocking os.path.exists
        mock_os_path_exists.return_value = False

        self.assertEqual(format_irq_as_str("20"), "Disabled interrupt")
        self.assertEqual(format_irq_as_str(""), "")

    def test_format_as_human(self):
        """Test the format_as_human function."""
        self.assertEqual(
            format_as_human("20231010123045"),
            datetime(2023, 10, 10, 12, 30, 45),
        )
        with self.assertRaises(ValueError):
            format_as_human("invalid_date")

    def test_format_as_seconds(self):
        """Test the format_as_seconds function."""
        self.assertEqual(
            format_as_seconds("20231010123045"),
            datetime(2023, 10, 10, 12, 30, 45).timestamp(),
        )
        with self.assertRaises(ValueError):
            format_as_seconds("invalid_date")

    def test_format_watts(self):
        """Test the format_watts function."""
        self.assertEqual(format_watts(12.3456), "12.35W")
        self.assertEqual(format_watts(0), "0.00W")

    def test_format_percent(self):
        """Test the format_percent function."""
        self.assertEqual(format_percent(12.3456), "12.35%")
        self.assertEqual(format_percent(0), "0.00%")

    def test_format_timedelta(self):
        """Test the format_timedelta function."""
        self.assertEqual(format_timedelta(3600), "1:00:00")
        self.assertEqual(format_timedelta(3661), "1:01:01")

    def test_get_energy_window(self):
        """Test the get_energy_window function."""
        # Snapshot timestamps win over the cycle duration
        rails = [(1, "RAIL", 0.0, 1.0, 1.0, 1000.0, 1030.5)]
        self.assertAlmostEqual(get_energy_window(rails, 33), 30.5)
        # Rows recorded before timestamps existed fall back to the duration
        self.assertEqual(get_energy_window([(1, "RAIL", 0.0, 1.0, 1.0)], 33), 33)
        self.assertEqual(
            get_energy_window([(1, "RAIL", 0.0, 1.0, 1.0, None, None)], 33), 33
        )
        self.assertEqual(
            get_energy_window([(1, "RAIL", 0.0, 1.0, 1.0, 1000.0, 1000.0)], 33), 33
        )
        self.assertEqual(get_energy_window([], 33), 33)

    def test_parse_hw_sleep(self):
        """Test the parse_hw_sleep function."""
        self.assertEqual(parse_hw_sleep(0.5), 50)
        self.assertEqual(parse_hw_sleep(1.0), 100)
        self.assertEqual(parse_hw_sleep(1.5), 0)


class TestSleepReport(unittest.TestCase):
    """Unit tests for the SleepReport class."""

    @patch("amd_debug.sleep_report.SleepDatabase")
    def setUp(self, MockSleepDatabase):
        """Set up a mock SleepReport instance for testing."""
        self.mock_db = MockSleepDatabase.return_value
        self.mock_db.report_summary_dataframe.return_value = pd.DataFrame(
            {
                "t0": [datetime(2023, 10, 10, 12, 0, 0).strftime("%Y%m%d%H%M%S")],
                "t1": [datetime(2023, 10, 10, 12, 30, 0).strftime("%Y%m%d%H%M%S")],
                "hw": [50],
                "requested": [1],
                "gpio": ["1, 2"],
                "wake_irq": ["1"],
                "b0": [90],
                "b1": [85],
                "full": [100],
            }
        )
        self.since = datetime(2023, 10, 9, 0, 0, 0)
        self.until = datetime(2023, 10, 12, 0, 0, 0)
        self.report = SleepReport(
            since=self.since,
            until=self.until,
            fname=None,
            fmt="txt",
            tool_debug=False,
            report_debug=False,
        )

    def test_analyze_duration(self):
        """Test the analyze_duration method."""
        self.report.analyze_duration(
            index=0,
            t0=datetime(2023, 10, 10, 12, 0, 0),
            t1=datetime(2023, 10, 10, 12, 30, 0),
            requested=20,
            hw=50,
        )
        self.assertEqual(len(self.report.failures), 2)

    def test_analyze_duration_residency_text(self):
        """The residency failure reports the window and percent sensibly."""
        self.report.failures = []
        self.report.analyze_duration(
            index=0,
            t0=datetime(2023, 10, 10, 12, 0, 0),
            t1=datetime(2023, 10, 10, 12, 1, 5),
            requested=60,
            hw=86.15,
            window=59.6,
        )
        self.assertEqual(len(self.report.failures), 1)
        text = self.report.failures[0][2]
        self.assertIn("asleep for 0:01:00", text)
        self.assertIn("86.15%", text)
        self.assertNotIn("8615", text)

    def test_analyze_duration_short_cycle(self):
        """Residency is only judged for cycles of at least 60 seconds."""
        self.report.failures = []
        self.report.analyze_duration(
            index=0,
            t0=datetime(2023, 10, 10, 12, 0, 0),
            t1=datetime(2023, 10, 10, 12, 0, 50),
            requested=20,
            hw=10,
            window=45,
        )
        self.assertEqual(len(self.report.failures), 0)

    @patch("amd_debug.sleep_report.Environment")
    @patch("amd_debug.sleep_report.FileSystemLoader")
    def test_build_template(self, _mock_fsl, mock_env):
        """Test the build_template method."""
        mock_template = mock_env.return_value.get_template.return_value
        mock_template.render.return_value = "Rendered Template"
        result = self.report.build_template(inc_prereq=False)
        self.assertEqual(result, "Rendered Template")

    @patch("amd_debug.sleep_report.Environment")
    @patch("amd_debug.sleep_report.FileSystemLoader")
    def test_build_template_fchown_called_while_fd_open(self, _mock_fsl, mock_env):
        """Test that fchown is called before the file descriptor is closed.

        When SUDO_UID/SUDO_GID are set, os.fchown must be called inside the
        'with os.fdopen(...)' block so that the file descriptor is still valid.
        Calling it after the block exits causes OSError (Bad file descriptor).
        """
        mock_template = mock_env.return_value.get_template.return_value
        mock_template.render.return_value = "Rendered Template"

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "report.txt")
            self.report.fname = fname

            with patch.dict("os.environ", {"SUDO_UID": "1000", "SUDO_GID": "1000"}):
                with patch("os.fchown") as mock_fchown:
                    self.report.build_template(inc_prereq=False)
                    # fchown must have been called exactly once with the real fd
                    mock_fchown.assert_called_once()
                    _fd, uid, gid = mock_fchown.call_args[0]
                    self.assertEqual(uid, 1000)
                    self.assertEqual(gid, 1000)

    @patch("matplotlib.pyplot.savefig")
    def test_build_battery_chart(self, mock_savefig):
        """Test the build_battery_chart method."""
        self.report.build_battery_chart()
        self.assertIsNotNone(self.report.battery_svg)
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    def test_build_hw_sleep_chart(self, mock_savefig):
        """Test the build_hw_sleep_chart method."""
        self.report.build_hw_sleep_chart()
        self.assertIsNotNone(self.report.hwsleep_svg)
        mock_savefig.assert_called_once()

    def test_pre_process_dataframe_zero_duration(self):
        """Test the pre_process_dataframe method when t0 and t1 are the same."""
        # Mock the dataframe with t0 and t1 being the same
        self.report.df = pd.DataFrame(
            {
                "t0": [datetime(2023, 10, 10, 12, 0, 0).strftime("%Y%m%d%H%M%S")],
                "t1": [datetime(2023, 10, 10, 12, 0, 0).strftime("%Y%m%d%H%M%S")],
                "hw": [50],
                "requested": [1],
                "gpio": ["1, 2"],
                "wake_irq": ["1"],
                "b0": [90],
                "b1": [85],
                "full": [100],
            }
        )

        # Call the method
        self.report.pre_process_dataframe()

        # Verify the dataframe was processed correctly
        self.assertTrue(
            self.report.df["Duration"].isna().iloc[0]
        )  # Duration should be NaN
        self.assertTrue(
            math.isnan(self.report.df["Hardware Sleep"].iloc[0])
        )  # Hardware Sleep should be NaN

    def test_battery_ave_rate(self):
        """Test the pre_process_dataframe Average Power calculation."""
        self.report.df = pd.DataFrame(
            {
                "t0": [datetime(2023, 10, 10, 12, 0, 0).strftime("%Y%m%d%H%M%S")],
                "t1": [datetime(2023, 10, 10, 13, 0, 0).strftime("%Y%m%d%H%M%S")],
                "hw": [50],
                "requested": [1],
                "gpio": ["1, 2"],
                "wake_irq": ["1"],
                "b0": [90000000],
                "b1": [85000000],
                "full": [100000000],
            }
        )
        self.report.pre_process_dataframe()
        batt_ave_rate = self.report.df["Average Power"].iloc[0]
        self.assertAlmostEqual(batt_ave_rate, -5, places=3)

    def test_battery_ave_rate_uses_snapshot_window(self):
        """Average Power from the battery divides by the snapshot window when present."""
        self.report.df = pd.DataFrame(
            {
                "t0": [datetime(2023, 10, 10, 12, 0, 0).strftime("%Y%m%d%H%M%S")],
                "t1": [datetime(2023, 10, 10, 13, 0, 0).strftime("%Y%m%d%H%M%S")],
                "hw": [50],
                "requested": [1],
                "gpio": ["1, 2"],
                "wake_irq": ["1"],
                "b0": [90000000],
                "b1": [85000000],
                "full": [100000000],
                "ts0": [1000.0],
                "ts1": [1000.0 + 1800],
            }
        )
        self.report.pre_process_dataframe()
        # 5 Wh over 30 minutes instead of over the 1 hour cycle
        self.assertAlmostEqual(self.report.df["Average Power"].iloc[0], -10, places=3)
        self.assertNotIn("ts0", self.report.df.columns)
        self.assertNotIn("ts1", self.report.df.columns)

    def test_hw_sleep_uses_snapshot_window(self):
        """Hardware sleep residency divides by the snapshot window when present."""
        self.report.failures = []
        self.report.df = pd.DataFrame(
            {
                "t0": [datetime(2023, 10, 10, 12, 0, 0).strftime("%Y%m%d%H%M%S")],
                "t1": [datetime(2023, 10, 10, 12, 1, 5).strftime("%Y%m%d%H%M%S")],
                "hw": [56],
                "requested": [60],
                "gpio": [""],
                "wake_irq": ["9"],
                "b0": [90000000],
                "b1": [85000000],
                "full": [100000000],
                "ts0": [1000.0],
                "ts1": [1000.0 + 59.6],
            }
        )
        self.report.pre_process_dataframe()
        # 56 s of hardware sleep in a 59.6 s window, not the 65 s cycle
        self.assertAlmostEqual(
            self.report.df["Hardware Sleep"].iloc[0], 56 / 59.6 * 100, places=3
        )
        self.assertEqual(self.report.failures, [])
        # The cycle duration shown in the summary is still the whole cycle
        self.assertEqual(self.report.df["Duration"].iloc[0], 65)

    def test_hw_sleep_falls_back_to_cycle_duration(self):
        """Cycles without snapshot timestamps use the whole cycle duration."""
        self.report.failures = []
        self.report.df = pd.DataFrame(
            {
                "t0": [datetime(2023, 10, 10, 12, 0, 0).strftime("%Y%m%d%H%M%S")],
                "t1": [datetime(2023, 10, 10, 12, 1, 5).strftime("%Y%m%d%H%M%S")],
                "hw": [56],
                "requested": [60],
                "gpio": [""],
                "wake_irq": ["9"],
                "b0": [90000000],
                "b1": [85000000],
                "full": [100000000],
                "ts0": [np.nan],
                "ts1": [np.nan],
            }
        )
        self.report.pre_process_dataframe()
        self.assertAlmostEqual(
            self.report.df["Hardware Sleep"].iloc[0], 56 / 65 * 100, places=3
        )
        self.assertEqual(len(self.report.failures), 1)
        self.assertIn("asleep for 0:01:05", self.report.failures[0][2])

    def test_calculate_power_rail_totals_uses_snapshot_window(self):
        """Rail power divides the energy delta by the snapshot window when present."""
        t0 = datetime(2023, 10, 10, 12, 0, 0)
        # 60 J (60000 mJ) over a 30 s snapshot window inside a 33 s cycle
        self.mock_db.report_power_rails.return_value = [
            (20231010120000, "SYS_IN", 0.0, 60000.0, 1.0, 1000.0, 1030.0),
        ]
        self.assertAlmostEqual(self.report.calculate_power_rail_totals(t0, 33), 2.0)
        summary = self.report.format_power_rail_data(t0, 33)
        self.assertIn("(over 30.0s)", summary)
        self.assertIn("SYS_IN: 2000.0mW", summary)

        # Rows without snapshot timestamps keep using the cycle duration
        self.mock_db.report_power_rails.return_value = [
            (20231010120000, "SYS_IN", 0.0, 66000.0, 1.0, None, None),
        ]
        self.assertAlmostEqual(self.report.calculate_power_rail_totals(t0, 33), 2.0)
        self.assertIn("(over 33.0s)", self.report.format_power_rail_data(t0, 33))

        # An unfinished cycle has no second snapshot
        self.mock_db.report_power_rails.return_value = [
            (20231010120000, "SYS_IN", 0.0, None, 1.0, 1000.0, None),
        ]
        self.assertIsNone(self.report.calculate_power_rail_totals(t0, 33))
        self.assertEqual(self.report.format_power_rail_data(t0, 33), "")

    def test_power_rails_ignore_list(self):
        """Ignored rails are listed but left out of the total and Average Power."""
        t0 = datetime(2023, 10, 10, 12, 0, 0)
        self.mock_db.report_power_rails.return_value = [
            (20231010120000, "SYS_IN", 0.0, 60000.0, 1.0, 1000.0, 1030.0),
            (20231010120000, "CPU_CORE", 0.0, 15000.0, 1.0, 1000.0, 1030.0),
            (20231010120000, "EDP", 0.0, 3000.0, 1.0, 1000.0, 1030.0),
        ]
        # Without an ignore list the feeder rail is double counted
        self.assertAlmostEqual(self.report.calculate_power_rail_totals(t0, 33), 2.6)

        self.report.ignore_rails = {"SYS_IN"}
        self.assertAlmostEqual(self.report.calculate_power_rail_totals(t0, 33), 0.6)
        summary = self.report.format_power_rail_data(t0, 33)
        self.assertIn("SYS_IN: 2000.0mW (ignored)", summary)
        self.assertIn("CPU_CORE: 500.0mW\n", summary)
        self.assertIn("Total: 600.0mW", summary)

        # Ignoring every rail leaves nothing to total
        self.report.ignore_rails = {"SYS_IN", "CPU_CORE", "EDP"}
        self.assertIsNone(self.report.calculate_power_rail_totals(t0, 33))

    @patch("amd_debug.sleep_report.print_color")
    def test_power_rails_ignore_list_summary(self, mock_print_color):
        """The summary's Average Power excludes ignored rails and unknown names warn."""
        self.mock_db.report_power_rails.return_value = [
            (20231010120000, "SYS_IN", 0.0, 60000.0, 1.0, 1000.0, 1030.0),
            (20231010120000, "CPU_CORE", 0.0, 15000.0, 1.0, 1000.0, 1030.0),
        ]
        self.report.ignore_rails = {"SYS_IN", "NOPE"}
        self.report.df = pd.DataFrame(
            {
                "t0": [datetime(2023, 10, 10, 12, 0, 0).strftime("%Y%m%d%H%M%S")],
                "t1": [datetime(2023, 10, 10, 12, 0, 33).strftime("%Y%m%d%H%M%S")],
                "hw": [30],
                "requested": [30],
                "gpio": [""],
                "wake_irq": ["9"],
                "b0": [90000000],
                "b1": [85000000],
                "full": [100000000],
            }
        )
        self.report.pre_process_dataframe()
        self.assertAlmostEqual(self.report.df["Average Power"].iloc[0], 0.5, places=3)
        mock_print_color.assert_called_once()
        self.assertIn("'NOPE'", mock_print_color.call_args[0][0])

    def test_get_prereq_data_preserves_markup_for_html_tables(self):
        """Ensure HTML prerequisite tables remain Markup and are not escaped."""
        self.report.format = "html"
        self.report.debug = True
        self.mock_db.get_last_prereq_ts.return_value = "20231010123045"
        self.mock_db.report_prereq.return_value = []
        table_text = "DMI|value\nfoo|bar"
        self.mock_db.report_debug.return_value = [(table_text, 6)]

        prereq, _t0, prereq_debug = self.report.get_prereq_data()

        self.assertEqual(prereq, [])
        self.assertEqual(len(prereq_debug), 1)
        self.assertIsInstance(prereq_debug[0]["data"], Markup)
        self.assertIn("<table", str(prereq_debug[0]["data"]))
        self.assertNotIn("&lt;table", str(prereq_debug[0]["data"]))
