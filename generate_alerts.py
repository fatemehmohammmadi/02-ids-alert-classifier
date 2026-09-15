"""Generate synthetic IDS-style alerts for supervised classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
OUT = Path(__file__).parent / "data" / "ids_alerts.csv"

SIGNATURES_BENIGN = [
    "policy-allowed-http",
    "dns-query-normal",
    "tls-handshake-ok",
    "icmp-echo-reply",
]
SIGNATURES_ATTACK = [
    "et-scan-portscan",
    "et-exploit-sql-injection",
    "et-malware-c2-beacon",
    "et-bruteforce-ssh",
    "et-dos-synflood",
]


def _rows(n: int, attack: bool) -> pd.DataFrame:
    sigs = SIGNATURES_ATTACK if attack else SIGNATURES_BENIGN
    severity = RNG.integers(3, 6, n) if attack else RNG.integers(1, 3, n)
    return pd.DataFrame(
        {
            "signature": RNG.choice(sigs, size=n),
            "proto": RNG.choice(["TCP", "UDP", "ICMP"], size=n, p=[0.75, 0.2, 0.05]),
            "src_bytes": (
                RNG.lognormal(12, 0.9, n).astype(int) if attack else RNG.lognormal(7.5, 0.6, n).astype(int)
            ),
            "dst_bytes": RNG.lognormal(8, 0.7, n).astype(int),
            "duration_s": RNG.uniform(0.01, 30, n).round(3),
            "failed_logins": RNG.integers(3, 40, n) if attack else RNG.integers(0, 2, n),
            "same_srv_rate": (
                RNG.uniform(0.6, 1.0, n).round(3) if attack else RNG.uniform(0.05, 0.5, n).round(3)
            ),
            "dst_host_count": RNG.integers(50, 500, n) if attack else RNG.integers(1, 40, n),
            "severity": severity,
            "label": "attack" if attack else "benign",
        }
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([_rows(700, False), _rows(300, True)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    df.insert(0, "alert_id", [f"A{i:05d}" for i in range(len(df))])
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} alerts -> {OUT}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
