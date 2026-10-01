# SPDX-License-Identifier: MIT

"""
Helpers for Intel platforms: ACPI LPIT residency counters and the
intel_pmc_core debugfs interface.

The kernel exposes two unprivileged counters derived from the ACPI LPIT table:
  * low_power_idle_cpu_residency_us    - time the package spent in PC10
  * low_power_idle_system_residency_us - time the SoC spent in S0ix (SLP_S0)

The intel_pmc_core driver additionally exposes (under debugfs, root only):
  * substate_residencies   - per S0ix.y substate residency counters
  * substate_requirements  - which IPs must be idle for each substate. Older
                             kernels add a Status column (Yes when the IP was
                             idle at the latched entry); kernels with S0ix
                             blocker counters (Lunar Lake and later) add a
                             Value column holding a per IP counter that
                             advances every time the IP blocked S0ix.
  * substate_status_registers - the latched per IP status bits (1 = idle)
  * package_cstate_show    - package C-state residency counters
  * ltr_show / pch_ip_power_gating_status - further blocker hints
"""

import logging
import os
import re

from amd_debug.common import read_file

LPIT_BASE = os.path.join("/", "sys", "devices", "system", "cpu", "cpuidle")
PMC_CORE_DEBUGFS = os.path.join("/", "sys", "kernel", "debug", "pmc_core")
PMC_CORE_PARAMS = os.path.join("/", "sys", "module", "intel_pmc_core", "parameters")

LPIT_CPU = "low_power_idle_cpu_residency_us"
LPIT_SYSTEM = "low_power_idle_system_residency_us"


def read_lpit_residency() -> dict:
    """Read the LPIT residency counters (in microseconds).

    Returns a dictionary with 'cpu' (PC10) and 'system' (S0ix) keys. A key is
    missing when the kernel doesn't expose the counter (no LPIT table).
    """
    result = {}
    for key, fname in [("cpu", LPIT_CPU), ("system", LPIT_SYSTEM)]:
        p = os.path.join(LPIT_BASE, fname)
        try:
            result[key] = int(read_file(p))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return result


def lpit_supported() -> bool:
    """Check whether the kernel exposes the LPIT system residency counter"""
    return os.path.exists(os.path.join(LPIT_BASE, LPIT_SYSTEM))


def parse_substate_residencies(text) -> dict:
    """Parse substate_residencies into {substate: residency_us}.

    Handles both known layouts:
        Substate   Residency
        S0i2.0     123
    and
        Substate   Residency   Enabled
        Enabled    S0i2.0      123
    """
    result = {}
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        name = None
        value = None
        for tok in tokens:
            if re.match(r"^S0i\d", tok):
                name = tok
            elif re.match(r"^\d+$", tok):
                value = int(tok)
        if name is not None and value is not None:
            result[name] = value
    return result


def parse_package_cstates(text) -> dict:
    """Parse package_cstate_show into {'Package C2': count, ...}"""
    result = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, value = line.rsplit(":", 1)
        name = name.strip()
        try:
            result[name] = int(value.strip())
        except ValueError:
            continue
    return result


def parse_substate_requirements(text) -> dict:
    """Parse substate_requirements.

    Older kernels report a latched status:
        Element    |   S0i2.0 |   S0i2.1 |   S0i2.2 |  Status |
        IP_FOO     | Required |          | Required |     Yes |
    Kernels with S0ix blocker counters report a per IP counter instead:
        Element    |   S0i2.0 |   S0i2.1 |   S0i2.2 |   Value |
        IP_FOO     | Required |          | Required |    1234 |
    Newer kernels prefix each PMC's table with a line naming the PMC.

    Returns {'modes': [...], 'format': 'status'|'counter'|'',
             'rows': [{'element', 'required', 'status', 'counter'}]}
    """
    modes = []
    rows = []
    fmt = ""
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        # drop the empty cell created by the trailing separator
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if len(cells) < 2:
            continue
        if cells[0] == "Element":
            modes = cells[1:-1]
            fmt = "counter" if cells[-1].lower() == "value" else "status"
            continue
        if not modes:
            continue
        required = set()
        for mode, cell in zip(modes, cells[1:-1]):
            if cell.lower() == "required":
                required.add(mode)
        counter = None
        if fmt == "counter":
            try:
                counter = int(cells[-1])
            except ValueError:
                counter = None
        rows.append(
            {
                # the kernel pads the element name ("pmc0: %34s")
                "element": " ".join(cells[0].split()),
                "required": required,
                "status": cells[-1],
                "counter": counter,
            }
        )
    return {"modes": modes, "format": fmt, "rows": rows}


def requirement_counters(requirements) -> dict:
    """Return {element: blocker counter} for a counter format requirements table"""
    result = {}
    for row in requirements.get("rows", []):
        if row.get("counter") is not None:
            result[row["element"]] = row["counter"]
    return result


def parse_substate_status(text) -> dict:
    """Parse substate_status_registers into {element: bit}.

    The file looks like:
        PMC0:LPM_STATUS_0:  0x333c1144
        PMC0:AON2_OFF_STS              0
        PMC0:AON4_OFF_STS              1
    A bit of 1 means the IP met its requirement when the status was latched.
    The element names are prefixed with 'pmcN: ' to match the requirements
    table of multi PMC kernels; both spellings are stored.
    """
    result = {}
    for line in text.splitlines():
        m = re.match(r"^\s*PMC(\d+):(\S+)\s+([01])\s*$", line)
        if not m:
            continue
        pmc, name, bit = m.groups()
        result[name] = int(bit)
        result[f"pmc{pmc}: {name}"] = int(bit)
    return result


def find_substate_blockers(requirements, mode, before=None, status=None) -> list:
    """Return IPs that are required for a substate but didn't report as idle.

    For a status format table the latched Yes/No column is used directly.
    For a counter format table the per IP blocker counters are cumulative
    since boot, so a pre-suspend snapshot (`before`) is needed: an IP blocked
    the cycle when its counter advanced. Without a snapshot the latched
    status bits from substate_status_registers (`status`) are used instead,
    and IPs that appear in neither are not reported.
    """
    blockers = []
    fmt = requirements.get("format", "status")
    for row in requirements.get("rows", []):
        if mode not in row["required"]:
            continue
        element = row["element"]
        if fmt == "counter":
            counter = row.get("counter")
            if before is not None and counter is not None and element in before:
                if counter - before[element] > 0:
                    blockers.append(element)
                continue
            if status is not None and element in status:
                if status[element] == 0:
                    blockers.append(element)
            continue
        if row["status"].lower() == "yes":
            continue
        blockers.append(element)
    return blockers


def substate_blocker_counts(requirements, mode, before) -> dict:
    """Return {element: counter delta} for the IPs reported by find_substate_blockers().

    Only meaningful for counter format tables with a pre-suspend snapshot. The
    PMC advances a counter for as long as the IP blocks an S0ix attempt, so a
    large delta is a chronic blocker and a small one a transient at entry.
    """
    result = {}
    if requirements.get("format") != "counter" or not before:
        return result
    for row in requirements.get("rows", []):
        element = row["element"]
        if mode not in row["required"] or row.get("counter") is None:
            continue
        if element in before and row["counter"] - before[element] > 0:
            result[element] = row["counter"] - before[element]
    return result


def blocker_method(requirements, before=None, status=None) -> str:
    """Describe how find_substate_blockers() decided on the blockers"""
    if requirements.get("format") != "counter":
        return "status latched"
    if before:
        return "blocker counter advanced during the cycle"
    if status:
        return "status latched"
    return ""


def substate_order(names) -> list:
    """Sort substate names from shallowest to deepest (S0i2.0 < S0i2.1 < S0i3.0)"""

    def key(name):
        nums = re.findall(r"\d+", name)
        return tuple(int(n) for n in nums)

    return sorted(names, key=key)


def diff_counters(before, after) -> dict:
    """Return the per key difference between two counter snapshots"""
    result = {}
    for name, value in after.items():
        if name not in before:
            continue
        result[name] = value - before[name]
    return result


class IntelPmcCore:
    """Thin wrapper around the intel_pmc_core debugfs and module parameters"""

    def __init__(self, debugfs=PMC_CORE_DEBUGFS, params=PMC_CORE_PARAMS):
        self.debugfs = debugfs
        self.params = params

    @property
    def available(self) -> bool:
        """Whether the pmc_core debugfs directory is present"""
        return os.path.isdir(self.debugfs)

    def read(self, name):
        """Read a debugfs file; returns None if it can't be read"""
        p = os.path.join(self.debugfs, name)
        try:
            return read_file(p)
        except FileNotFoundError:
            return None
        except PermissionError:
            logging.debug("Unable to read %s", p)
            return None
        except OSError as e:
            logging.debug("Failed to read %s: %s", p, e)
            return None

    def write(self, name, value) -> bool:
        """Write to a debugfs file; returns False if it can't be written"""
        p = os.path.join(self.debugfs, name)
        try:
            with open(p, "w", encoding="utf-8") as w:
                w.write(value)
        except (FileNotFoundError, PermissionError, OSError) as e:
            logging.debug("Failed to write %s to %s: %s", value, p, e)
            return False
        return True

    def get_param(self, name):
        """Read a module parameter; returns None if unavailable"""
        p = os.path.join(self.params, name)
        try:
            return read_file(p)
        except (FileNotFoundError, PermissionError):
            return None

    def set_param(self, name, value) -> bool:
        """Write a module parameter; returns False if it can't be written"""
        p = os.path.join(self.params, name)
        try:
            with open(p, "w", encoding="utf-8") as w:
                w.write(value)
        except (FileNotFoundError, PermissionError, OSError) as e:
            logging.debug("Failed to write %s to %s: %s", value, p, e)
            return False
        return True

    def latch_mode(self, mode) -> bool:
        """Latch the substate requirement status on entry to `mode`.

        `mode` is either 'c10' (package C10 entry) or a substate name such as
        'S0i2.0'. Previously latched events are cleared first.
        """
        if not self.write("lpm_latch_mode", "clear"):
            return False
        return self.write("lpm_latch_mode", mode)

    def substate_residencies(self) -> dict:
        """Read and parse the substate residency counters"""
        text = self.read("substate_residencies")
        if text is None:
            return {}
        return parse_substate_residencies(text)

    def package_cstates(self) -> dict:
        """Read and parse the package C-state counters"""
        text = self.read("package_cstate_show")
        if text is None:
            return {}
        return parse_package_cstates(text)

    def substate_requirements(self) -> dict:
        """Read and parse the substate requirements table"""
        text = self.read("substate_requirements")
        if text is None:
            return {}
        return parse_substate_requirements(text)

    def substate_status(self) -> dict:
        """Read and parse the latched substate status registers"""
        text = self.read("substate_status_registers")
        if text is None:
            return {}
        return parse_substate_status(text)


STATE_FILE = os.path.join("/", "var", "lib", "amd-s2idle", "intel-cycle.json")


def save_cycle_state(state, fname=STATE_FILE) -> bool:
    """Persist pre-suspend counter snapshots for the systemd hook path.

    The systemd pre and post hooks run in separate processes, so the counter
    snapshots taken before suspend need to survive until after resume.
    """
    import json  # pylint: disable=import-outside-toplevel

    try:
        with open(fname, "w", encoding="utf-8") as w:
            json.dump(state, w)
    except OSError as e:
        logging.debug("Failed to save Intel cycle state to %s: %s", fname, e)
        return False
    return True


def load_cycle_state(fname=STATE_FILE) -> dict:
    """Load and remove the pre-suspend counter snapshots"""
    import json  # pylint: disable=import-outside-toplevel

    try:
        with open(fname, "r", encoding="utf-8") as r:
            state = json.load(r)
    except (OSError, ValueError) as e:
        logging.debug("No Intel cycle state in %s: %s", fname, e)
        return {}
    try:
        os.unlink(fname)
    except OSError:
        pass
    if not isinstance(state, dict):
        return {}
    return state
