"""Re-derives the sha256 of each pre-registration JSON from its own contents and checks it against the stored value. this is the pre-registration integrity check: the hash covers every field except sha256 itself, computed as sha256 of json.dumps(payload, sort_keys=True), so a reviewer can confirm the frozen sample wasn't altered after the fact without trusting anything but the committed file. also confirms the supplement's excluded_primary_sha points back at the primary."""

from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

PRIMARY = "p3_preregistration.json"
SUPPLEMENT = "p3_oversample_true_drift.json"


def recompute(path: str | Path) -> tuple[str, str]:
    payload = json.loads(Path(path).read_text())
    stored = payload.pop("sha256")
    derived = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return stored, derived


def main(results_dir: str | Path = ".") -> int:
    d = Path(results_dir)
    ok = True

    for name in (PRIMARY, SUPPLEMENT):
        p = d / name
        if not p.exists():
            print(f"[HASH] MISSING {name}")
            ok = False
            continue
        stored, derived = recompute(p)
        match = stored == derived
        ok &= match
        print(f"[HASH] {name}")
        print(f"       stored  {stored}")
        print(f"       derived {derived}")
        print(f"       {'MATCH' if match else 'MISMATCH'}")

    primary = d / PRIMARY
    supp = d / SUPPLEMENT
    if primary.exists() and supp.exists():
        p_sha = json.loads(primary.read_text())["sha256"]
        linked = json.loads(supp.read_text()).get("excluded_primary_sha")
        chain = p_sha == linked
        ok &= chain
        print(f"[HASH] supplement excluded_primary_sha links to primary: "
              f"{'YES' if chain else 'NO'}")

    print(f"\n[HASH] {'all checks passed' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
