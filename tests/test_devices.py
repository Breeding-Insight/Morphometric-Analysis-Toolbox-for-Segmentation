"""BiRefNet device reporting without requiring real GPU hardware."""

import sys
from types import SimpleNamespace

from mats import devices


class _Tensor:
    def __add__(self, _other):
        return self

    def sum(self):
        return self

    def item(self):
        return 1


def _torch(cuda=False, mps=False):
    return SimpleNamespace(
        __version__="test",
        ones=lambda *_args, **_kwargs: _Tensor(),
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            synchronize=lambda: None,
            get_device_name=lambda _device: "Test CUDA",
        ),
        mps=SimpleNamespace(synchronize=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_built=lambda: mps, is_available=lambda: mps)),
    )


def test_cuda_report_is_green(monkeypatch):
    monkeypatch.delenv("BIREFNET_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _torch(cuda=True, mps=True))
    report = devices.birefnet_device_report()
    assert (report.device, report.severity, report.accelerated) == ("cuda", "success", True)


def test_cpu_fallback_is_yellow(monkeypatch):
    monkeypatch.delenv("BIREFNET_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _torch())
    report = devices.birefnet_device_report()
    assert (report.device, report.severity, report.accelerated) == ("cpu", "warning", False)


def test_invalid_forced_device_is_red(monkeypatch):
    monkeypatch.setenv("BIREFNET_DEVICE", "banana")
    monkeypatch.setitem(sys.modules, "torch", _torch(cuda=True))
    report = devices.birefnet_device_report()
    assert report.severity == "error"


def test_available_cpu_workers_respects_scheduler_and_affinity(monkeypatch):
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NCPUS", "NSLOTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(devices.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(devices.os, "sched_getaffinity", lambda _pid: set(range(12)), raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")

    assert devices.available_cpu_workers() == 4


def test_available_cpu_workers_ignores_invalid_scheduler_values(monkeypatch):
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NCPUS", "NSLOTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(devices.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(devices.os, "sched_getaffinity", lambda _pid: set(range(8)), raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")

    assert devices.available_cpu_workers() == 6


def test_worker_risk_bands_and_break_glass_threshold():
    assert devices.worker_risk_report(2, 8).level == "low"
    assert devices.worker_risk_report(4, 8).level == "moderate"
    assert devices.worker_risk_report(6, 8).level == "high"

    critical = devices.worker_risk_report(7, 8)
    assert critical.level == "critical"
    assert critical.normal_limit == 6
    assert critical.requires_break_glass is True


def test_single_worker_capacity_does_not_require_break_glass():
    report = devices.worker_risk_report(1, 1)

    assert report.level == "limited"
    assert report.requires_break_glass is False
