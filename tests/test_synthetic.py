from __future__ import annotations

from aiperfopt.synthetic import (
    HELD_OUT_SCENARIO_GENERATOR,
    HELD_OUT_SCENARIO_NAME,
    SCENARIO_GENERATORS,
    generate_cpu_stress_scenario,
    generate_idle_scenario,
)


def test_generators_are_deterministic_given_seed():
    a = generate_idle_scenario(duration_seconds=100, interval_seconds=2.0, seed=42)
    b = generate_idle_scenario(duration_seconds=100, interval_seconds=2.0, seed=42)
    assert [s.cpu_percent for s in a] == [s.cpu_percent for s in b]


def test_different_seeds_produce_different_noise():
    a = generate_idle_scenario(duration_seconds=100, interval_seconds=2.0, seed=1)
    b = generate_idle_scenario(duration_seconds=100, interval_seconds=2.0, seed=2)
    assert [s.cpu_percent for s in a] != [s.cpu_percent for s in b]


def test_cpu_percent_stays_within_valid_percentage_range():
    for generator in list(SCENARIO_GENERATORS.values()) + [HELD_OUT_SCENARIO_GENERATOR]:
        samples = generator(duration_seconds=300, interval_seconds=2.0)
        for s in samples:
            assert 0.0 <= s.cpu_percent <= 100.0
            assert 0.0 <= s.mem_percent <= 100.0


def test_counters_are_monotonically_increasing():
    """Synthetic disk/net counters must look like real psutil cumulative
    counters (never decrease), since Feature Engineering's rate transform
    treats a decrease as a counter reset."""
    samples = generate_cpu_stress_scenario(duration_seconds=300, interval_seconds=2.0)
    disk_read = [s.disk_read_bytes for s in samples]
    net_sent = [s.net_bytes_sent for s in samples]
    assert all(b >= a for a, b in zip(disk_read, disk_read[1:]))
    assert all(b >= a for a, b in zip(net_sent, net_sent[1:]))


def test_cpu_stress_scenario_shows_a_clear_ramp():
    samples = generate_cpu_stress_scenario(duration_seconds=900, interval_seconds=2.0)
    early_cpu = [s.cpu_percent for s in samples[:20]]
    late_cpu = [s.cpu_percent for s in samples[-20:]]
    assert sum(late_cpu) / len(late_cpu) > sum(early_cpu) / len(early_cpu) + 30


def test_regular_sampling_interval():
    samples = generate_idle_scenario(duration_seconds=100, interval_seconds=5.0)
    diffs = [b.timestamp - a.timestamp for a, b in zip(samples, samples[1:])]
    assert all(abs(d - 5.0) < 1e-9 for d in diffs)


def test_start_timestamp_offsets_the_whole_series():
    samples = generate_idle_scenario(duration_seconds=50, interval_seconds=2.0, start_timestamp=100_000.0)
    assert samples[0].timestamp == 100_000.0


def test_scenario_registry_excludes_held_out_scenario():
    assert HELD_OUT_SCENARIO_NAME not in SCENARIO_GENERATORS
    assert len(SCENARIO_GENERATORS) >= 3
