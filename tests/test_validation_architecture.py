import ast
from pathlib import Path


def test_production_package_does_not_import_skeleton_lab():
    root = Path(__file__).parents[1] / "prism"
    production = [path for path in root.rglob("*.py") if "validation" not in path.parts]
    offenders = []
    for path in production:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import) and any(alias.name.split('.')[0] == 'skeleton_lab' for alias in node.names):
                offenders.append(path)
            if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'skeleton_lab':
                offenders.append(path)
    assert offenders == []


def test_source_has_no_type_annotations():
    root = Path(__file__).parents[1]
    offenders = []
    for folder in ("prism", "scripts", "tests"):
        for path in (root / folder).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign):
                    offenders.append((path, node.lineno))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    annotations = [node.returns]
                    annotations.extend(
                        argument.annotation
                        for argument in (
                            *node.args.posonlyargs,
                            *node.args.args,
                            *node.args.kwonlyargs,
                        )
                    )
                    if node.args.vararg:
                        annotations.append(node.args.vararg.annotation)
                    if node.args.kwarg:
                        annotations.append(node.args.kwarg.annotation)
                    if any(value is not None for value in annotations):
                        offenders.append((path, node.lineno))
    assert offenders == []


def test_matplotlib_is_confined_to_plotting_package():
    root = Path(__file__).parents[1]
    offenders = []
    for folder in ("prism", "scripts"):
        for path in (root / folder).rglob("*.py"):
            if "plotting" in path.parts:
                continue
            text = path.read_text()
            if "matplotlib" in text or "plt." in text or "savefig" in text:
                offenders.append(path)
    assert offenders == []


def test_accelerometer_validation_is_not_packaged():
    package = Path(__file__).parents[1] / "prism"
    names = {path.name for path in package.rglob("*.py")}
    assert "run_accelerometer_validation.py" not in names
    assert "skeleton_accelerometer_validation.py" not in names
