#!/usr/bin/python3
# SPDX-License-Identifier: MIT

"""
This module contains unit tests for the datbase functions in the amd-debug-tools package.
"""
import os
import sqlite3
import tempfile
import unittest

from datetime import datetime
from unittest.mock import patch, MagicMock

from amd_debug.database import SleepDatabase


class TestSleepDatabase(unittest.TestCase):
    """Test SleepDatabase class"""

    def setUp(self):
        """Set up mocks and an in-memory database for testing"""
        # Initialize SleepDatabase after mocks are set up
        self.db = SleepDatabase(dbf=":memory:")

    def test_start_cycle(self):
        """Test starting a new sleep cycle"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.assertEqual(self.db.last_suspend, timestamp)
        self.assertEqual(self.db.cycle_data_cnt, 0)
        self.assertEqual(self.db.debug_cnt, 0)

    def test_record_debug(self):
        """Test recording a debug message"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_debug("Test debug message", level=5)
        cur = self.db.db.cursor()
        cur.execute(
            "SELECT message, priority FROM debug WHERE t0=?",
            (int(timestamp.strftime("%Y%m%d%H%M%S")),),
        )
        result = cur.fetchone()
        self.assertEqual(result, ("Test debug message", 5))

    def test_record_battery_energy(self):
        """Test recording battery energy"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_battery_energy("Battery1", 50, 100, "mWh")
        cur = self.db.db.cursor()
        cur.execute(
            "SELECT name, b0, b1, full, unit FROM battery WHERE t0=?",
            (int(timestamp.strftime("%Y%m%d%H%M%S")),),
        )
        result = cur.fetchone()
        self.assertEqual(result, ("Battery1", 50, None, 100, "mWh"))

    def test_record_cycle_data(self):
        """Test recording cycle data"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_cycle_data("Test cycle data", "symbol1")
        cur = self.db.db.cursor()
        cur.execute(
            "SELECT message, symbol FROM cycle_data WHERE t0=?",
            (int(timestamp.strftime("%Y%m%d%H%M%S")),),
        )
        result = cur.fetchone()
        self.assertEqual(result, ("Test cycle data", "symbol1"))

    def test_record_cycle(self):
        """Test recording a sleep cycle"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_cycle(
            requested_duration=100,
            active_gpios="GPIO1",
            wakeup_irqs="IRQ1",
            kernel_duration=1.5,
            hw_sleep_duration=2.5,
        )
        cur = self.db.db.cursor()
        cur.execute(
            "SELECT requested, gpio, wake_irq, kernel, hw FROM cycle WHERE t0=?",
            (int(timestamp.strftime("%Y%m%d%H%M%S")),),
        )
        result = cur.fetchone()
        self.assertEqual(result, (100, "GPIO1", "IRQ1", 1.5, 2.5))

    def test_report_debug(self):
        """Test reporting debug messages"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_debug("Test debug message", level=5)
        result = self.db.report_debug(timestamp)
        self.assertEqual(result, [("Test debug message", 5)])

    def test_report_battery(self):
        """Test reporting battery data"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_battery_energy("Battery1", 50, 100, "mWh", timestamp=1000.0)
        result = self.db.report_battery(timestamp)
        self.assertEqual(
            result,
            [
                (
                    int(timestamp.strftime("%Y%m%d%H%M%S")),
                    "Battery1",
                    50,
                    None,
                    100,
                    "mWh",
                    1000.0,
                    None,
                )
            ],
        )

    def test_record_battery_energy_timestamps(self):
        """Test that both battery snapshots record when they were taken"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_battery_energy("Battery1", 50, 100, "mWh", timestamp=1000.0)
        self.db.record_battery_energy("Battery1", 40, 100, "mWh", timestamp=1030.5)
        result = self.db.report_battery(timestamp)
        self.assertEqual(result[0][2:], (50, 40, 100, "mWh", 1000.0, 1030.5))

    @patch("amd_debug.database.time.time", return_value=12345.5)
    def test_record_battery_energy_default_timestamp(self, _mock_time):
        """Test that the battery snapshot timestamp defaults to now"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_battery_energy("Battery1", 50, 100, "mWh")
        result = self.db.report_battery(timestamp)
        self.assertEqual(result[0][6], 12345.5)

    def test_record_prereq(self):
        """Test recording a prereq message"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_prereq("Test prereq message", "symbol1")
        cur = self.db.db.cursor()
        cur.execute(
            "SELECT message, symbol FROM prereq_data WHERE t0=?",
            (int(timestamp.strftime("%Y%m%d%H%M%S")),),
        )
        result = cur.fetchone()
        self.assertEqual(result, ("Test prereq message", "symbol1"))

    def test_report_prereq(self):
        """Test reporting prereq messages"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_prereq("Test prereq message 1", "symbol1")
        self.db.record_prereq("Test prereq message 2", "symbol2")
        result = self.db.report_prereq(timestamp)
        self.assertEqual(
            result,
            [
                (
                    int(timestamp.strftime("%Y%m%d%H%M%S")),
                    0,
                    "Test prereq message 1",
                    "symbol1",
                ),
                (
                    int(timestamp.strftime("%Y%m%d%H%M%S")),
                    1,
                    "Test prereq message 2",
                    "symbol2",
                ),
            ],
        )

    def test_report_prereq_no_data(self):
        """Test reporting prereq messages when no data exists"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        result = self.db.report_prereq(timestamp)
        self.assertEqual(result, [])

    def test_report_prereq_none_timestamp(self):
        """Test reporting prereq messages with None timestamp"""
        result = self.db.report_prereq(None)
        self.assertEqual(result, [])

    def test_report_cycle(self):
        """Test reporting a cycle from the database"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_cycle(
            requested_duration=100,
            active_gpios="GPIO1",
            wakeup_irqs="IRQ1",
            kernel_duration=1.5,
            hw_sleep_duration=2.5,
        )
        result = self.db.report_cycle(timestamp)
        self.assertEqual(
            result,
            [
                (
                    int(timestamp.strftime("%Y%m%d%H%M%S")),
                    int(datetime.now().strftime("%Y%m%d%H%M%S")),
                    100,
                    "GPIO1",
                    "IRQ1",
                    1.5,
                    2.5,
                )
            ],
        )

    def test_report_cycle_no_data(self):
        """Test reporting a cycle when no data exists"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        result = self.db.report_cycle(timestamp)
        self.assertEqual(result, [])

    def test_report_cycle_none_timestamp(self):
        """Test reporting a cycle with None timestamp"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_cycle(
            requested_duration=100,
            active_gpios="GPIO1",
            wakeup_irqs="IRQ1",
            kernel_duration=1.5,
            hw_sleep_duration=2.5,
        )
        result = self.db.report_cycle(None)
        self.assertEqual(
            result,
            [
                (
                    int(timestamp.strftime("%Y%m%d%H%M%S")),
                    int(datetime.now().strftime("%Y%m%d%H%M%S")),
                    100,
                    "GPIO1",
                    "IRQ1",
                    1.5,
                    2.5,
                )
            ],
        )

    def test_get_last_cycle(self):
        """Test getting the last cycle from the database"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_cycle(
            requested_duration=100,
            active_gpios="GPIO1",
            wakeup_irqs="IRQ1",
            kernel_duration=1.5,
            hw_sleep_duration=2.5,
        )
        result = self.db.get_last_cycle()
        self.assertEqual(result, (int(timestamp.strftime("%Y%m%d%H%M%S")),))

    def test_get_last_cycle_no_data(self):
        """Test getting the last cycle when no data exists"""
        result = self.db.get_last_cycle()
        self.assertIsNone(result)

    def test_get_last_prereq_ts(self):
        """Test getting the last prereq timestamp from the database"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_prereq("Test prereq message 1", "symbol1")
        self.db.record_prereq("Test prereq message 2", "symbol2")
        result = self.db.get_last_prereq_ts()
        self.assertEqual(result, int(timestamp.strftime("%Y%m%d%H%M%S")))

    def test_get_last_prereq_ts_no_data(self):
        """Test getting the last prereq timestamp when no data exists"""
        result = self.db.get_last_prereq_ts()
        self.assertEqual(result, 0)

    def test_report_cycle_data(self):
        """Test reporting cycle data"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_cycle_data("Test cycle data 1", "symbol1")
        self.db.record_cycle_data("Test cycle data 2", "symbol2")
        result = self.db.report_cycle_data(timestamp)
        expected_result = "symbol1 Test cycle data 1\nsymbol2 Test cycle data 2\n"
        self.assertEqual(result, expected_result)

    def test_report_cycle_data_no_data(self):
        """Test reporting cycle data when no data exists"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        result = self.db.report_cycle_data(timestamp)
        self.assertEqual(result, "")

    def test_report_cycle_data_none_timestamp(self):
        """Test reporting cycle data with None timestamp"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_cycle_data("Test cycle data 1", "symbol1")
        result = self.db.report_cycle_data(None)
        expected_result = "symbol1 Test cycle data 1\n"
        self.assertEqual(result, expected_result)

    def test_record_power_rail_energy_insert(self):
        """Test inserting power rail energy (first call in prep)"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_power_rail_energy("CPU_VDDCR_PH1", 1000000.0, 149011.611)
        cur = self.db.db.cursor()
        cur.execute(
            "SELECT label, e0, e1, scale FROM power_rails WHERE t0=?",
            (int(timestamp.strftime("%Y%m%d%H%M%S")),),
        )
        result = cur.fetchone()
        self.assertEqual(result, ("CPU_VDDCR_PH1", 1000000.0, None, 149011.611))

    def test_record_power_rail_energy_update(self):
        """Test updating power rail energy (second call in post)"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        # First call (prep) - INSERT
        self.db.record_power_rail_energy("CPU_VDDCR_PH1", 1000000.0, 149011.611)
        # Second call (post) - UPDATE
        self.db.record_power_rail_energy("CPU_VDDCR_PH1", 1050000.0, 149011.611)
        cur = self.db.db.cursor()
        cur.execute(
            "SELECT label, e0, e1, scale FROM power_rails WHERE t0=?",
            (int(timestamp.strftime("%Y%m%d%H%M%S")),),
        )
        result = cur.fetchone()
        self.assertEqual(result, ("CPU_VDDCR_PH1", 1000000.0, 1050000.0, 149011.611))

    def test_record_multiple_power_rails(self):
        """Test recording multiple power rails per cycle"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        # Record multiple rails
        rails = [
            ("CPU_VDDCR_PH1", 1000000.0, 1050000.0, 149011.611),
            ("CPU_VDDCR_PH2", 2000000.0, 2100000.0, 149011.611),
            ("VDDIO", 3000000.0, 3150000.0, 23751.3),
        ]
        for label, e0, e1, scale in rails:
            self.db.record_power_rail_energy(label, e0, scale)
            self.db.record_power_rail_energy(label, e1, scale)

        cur = self.db.db.cursor()
        cur.execute(
            "SELECT label, e0, e1, scale FROM power_rails WHERE t0=? ORDER BY label",
            (int(timestamp.strftime("%Y%m%d%H%M%S")),),
        )
        results = cur.fetchall()
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], ("CPU_VDDCR_PH1", 1000000.0, 1050000.0, 149011.611))
        self.assertEqual(results[1], ("CPU_VDDCR_PH2", 2000000.0, 2100000.0, 149011.611))
        self.assertEqual(results[2], ("VDDIO", 3000000.0, 3150000.0, 23751.3))

    def test_report_power_rails(self):
        """Test reporting power rails"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_power_rail_energy(
            "CPU_VDDCR_PH1", 1000000.0, 149011.611, timestamp=1000.0
        )
        self.db.record_power_rail_energy(
            "CPU_VDDCR_PH1", 1050000.0, 149011.611, timestamp=1030.5
        )
        result = self.db.report_power_rails(timestamp)
        self.assertEqual(
            result,
            [
                (
                    int(timestamp.strftime("%Y%m%d%H%M%S")),
                    "CPU_VDDCR_PH1",
                    1000000.0,
                    1050000.0,
                    149011.611,
                    1000.0,
                    1030.5,
                )
            ],
        )

    @patch("amd_debug.database.time.time", return_value=12345.5)
    def test_record_power_rail_energy_default_timestamp(self, _mock_time):
        """Test that the power rail snapshot timestamp defaults to now"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        self.db.record_power_rail_energy("CPU_VDDCR_PH1", 1000000.0, 149011.611)
        result = self.db.report_power_rails(timestamp)
        self.assertEqual(result[0][5:], (12345.5, None))

    def test_report_power_rails_no_data(self):
        """Test reporting power rails when no data exists"""
        timestamp = datetime.now()
        self.db.start_cycle(timestamp)
        result = self.db.report_power_rails(timestamp)
        self.assertEqual(result, [])

    def test_default_db_path_uses_var_lib_restrictive(self):
        """Test the default database path is created securely in /var/lib"""
        with patch("amd_debug.database.os.makedirs") as makedirs, patch(
            "amd_debug.database.os.path.exists", return_value=False
        ), patch("amd_debug.database.sqlite3.connect", return_value=MagicMock()):
            SleepDatabase()

            # Directory is created in /var/lib with a restrictive mode
            makedirs.assert_called_once_with(
                "/var/lib/amd-s2idle", mode=0o700, exist_ok=True
            )

            # The insecure /var/local fallback is never referenced
            for call in makedirs.call_args_list:
                for arg in call.args:
                    self.assertNotIn("/var/local", str(arg))

    def test_schema_migration_v1_to_v2(self):
        """Test database migration from schema v1 to v2"""
        # Create a new database (will be at v2)
        db = SleepDatabase(dbf=":memory:")
        cur = db.db.cursor()

        # Verify power_rails table exists
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='power_rails'"
        )
        result = cur.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "power_rails")

        # Verify schema version is current
        cur.execute("PRAGMA user_version")
        version = cur.fetchone()[0]
        self.assertEqual(version, 3)

    def _columns(self, db, table):
        cur = db.db.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cur.fetchall()]

    def test_schema_migration_v2_to_v3(self):
        """Test that a v2 database gains the energy snapshot timestamp columns"""
        with tempfile.TemporaryDirectory() as tmp:
            dbf = os.path.join(tmp, "data.db")
            con = sqlite3.connect(dbf)
            cur = con.cursor()
            cur.execute(
                "CREATE TABLE battery (t0 INTEGER PRIMARY KEY, name TEXT, b0 INTEGER, "
                "b1 INTEGER, full INTEGER, unit TEXT)"
            )
            cur.execute(
                "CREATE TABLE power_rails (t0 INTEGER, label TEXT, e0 REAL, e1 REAL, "
                "scale REAL, PRIMARY KEY(t0, label))"
            )
            cur.execute("INSERT INTO battery VALUES (1, 'BAT0', 10, 5, 100, 'W')")
            cur.execute("PRAGMA user_version = 2")
            con.commit()
            con.close()

            db = SleepDatabase(dbf=dbf)
            for table in ("battery", "power_rails"):
                self.assertIn("ts0", self._columns(db, table))
                self.assertIn("ts1", self._columns(db, table))
            cur = db.db.cursor()
            cur.execute("PRAGMA user_version")
            self.assertEqual(cur.fetchone()[0], 3)
            # Old rows survive with no timestamps
            cur.execute("SELECT b0, b1, ts0, ts1 FROM battery")
            self.assertEqual(cur.fetchone(), (10, 5, None, None))
            del db

    def test_schema_migration_v0_chains(self):
        """Test that a v0 database receives every migration step"""
        with tempfile.TemporaryDirectory() as tmp:
            dbf = os.path.join(tmp, "data.db")
            con = sqlite3.connect(dbf)
            cur = con.cursor()
            cur.execute(
                "CREATE TABLE debug (t0 INTEGER, id INTEGER, message TEXT, "
                "PRIMARY KEY(t0, id))"
            )
            cur.execute(
                "CREATE TABLE battery (t0 INTEGER PRIMARY KEY, name TEXT, b0 INTEGER, "
                "b1 INTEGER, full INTEGER, unit TEXT)"
            )
            cur.execute("PRAGMA user_version = 0")
            con.commit()
            con.close()

            db = SleepDatabase(dbf=dbf)
            self.assertIn("priority", self._columns(db, "debug"))
            self.assertIn("ts1", self._columns(db, "battery"))
            self.assertIn("ts1", self._columns(db, "power_rails"))
            cur = db.db.cursor()
            cur.execute("PRAGMA user_version")
            self.assertEqual(cur.fetchone()[0], 3)
            del db
