from __future__ import annotations

from pathlib import Path

from reward_sanity import evaluate_sanity, load_sanity_config

ROOT = Path(__file__).resolve().parents[2]


def test_reward_sanity_config_covers_anti_hacking_and_gate_cases() -> None:
    config = load_sanity_config(ROOT / "configs" / "matrices" / "reward_sanity.yaml")

    assert {item.name for item in config.cases} == {
        "cruise",
        "stationary",
        "extremely_low_speed",
        "slower_progress",
        "low_route_progress",
        "overspeed",
        "following_non_closing",
        "approaching_collision",
        "uncomfortable",
        "collision",
        "out_of_road",
        "wrong_direction",
    }
    assert len(config.comparisons) == 7


def test_reward_sanity_report_requires_every_declared_check_to_pass() -> None:
    config = load_sanity_config(ROOT / "configs" / "matrices" / "reward_sanity.yaml")

    report = evaluate_sanity(config)

    assert report["status"] == "passed"
    assert report["case_count"] == 12
    assert all(item["passed"] for item in report["checks"])
