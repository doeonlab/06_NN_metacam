import importlib


def test_core_imports():
    modules = [
        "metacam",
        "metacam.physics.propagation",
        "metacam.data.io",
        "metacam.ops.torch_ops",
        "metacam.metrics.losses",
        "Library.fieldprop",
        "fieldprop.fieldprop",
    ]
    for module in modules:
        importlib.import_module(module)
