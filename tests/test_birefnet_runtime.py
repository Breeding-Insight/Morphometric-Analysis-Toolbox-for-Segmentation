"""Local-only BiRefNet loading behavior without real model weights."""

import sys
import types

import pytest


def test_runtime_reports_missing_dependencies(monkeypatch):
    from mats import birefnet_runtime

    real_import = birefnet_runtime.importlib.import_module

    def fake_import(name):
        if name in {"kornia", "timm"}:
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(birefnet_runtime.importlib, "import_module", fake_import)
    status = birefnet_runtime.birefnet_runtime_status()

    assert status.ready is False
    assert status.missing == ("kornia", "timm")
    assert "python -m pip install" in status.detail


def test_local_loader_never_calls_auto_fetch_or_transformers(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    from mats import core, weights

    class FakeModel:
        def __init__(self):
            self.loaded = None
            self.device = None
            self.is_eval = False

        def load_state_dict(self, state_dict, strict):
            self.loaded = (state_dict, strict)

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.is_eval = True
            return self

    fake_model = FakeModel()
    fake_runtime = types.ModuleType("mats.birefnet_runtime")
    fake_runtime.require_birefnet_dependencies = lambda: None
    fake_model_module = types.ModuleType("mats.models.birefnet")
    fake_model_module.create_birefnet_model = lambda: fake_model

    monkeypatch.setitem(sys.modules, "mats.birefnet_runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "mats.models.birefnet", fake_model_module)
    monkeypatch.setattr(weights, "require_local_weight", lambda name: tmp_path / "birefnet_leaf.pth")
    monkeypatch.setattr(weights, "ensure_weight", lambda name: pytest.fail("must not auto-fetch"))
    monkeypatch.setattr(core.torch, "load", lambda *args, **kwargs: {"model_state_dict": {"local": 1}})
    monkeypatch.setattr(core, "resolve_birefnet_device", lambda *_: core.torch.device("cpu"))
    core._BIREFNET_MODELS.clear()

    model = core.get_birefnet_model()

    assert model is fake_model
    assert fake_model.loaded == ({"local": 1}, True)
    assert str(fake_model.device) == "cpu"
    assert fake_model.is_eval is True
