#!/usr/bin/env python3
"""Build a combined Comprehensive Head Classification ONNX model."""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import ModelProto, ValueInfoProto, helper


TARGET_IR_VERSION = 8
TARGET_ONNX_OPSET = 16


@dataclass(frozen=True)
class BranchSpec:
    name: str
    model_path: Path
    source_input: str
    public_input: str
    source_output: str
    public_output: str
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CHC branch ONNX files into chc_{variant}.onnx."
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Model variant, such as l, n, p, or s.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output ONNX path. Defaults to chc_{variant}.onnx, "
            "or chc_{variant}_wo_fiqa.onnx with --disable-fiqa."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("."),
        help="Directory containing source ONNX files. Defaults to the current directory.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare merged outputs with the source models using deterministic random inputs.",
    )
    parser.add_argument(
        "--disable-fiqa",
        action="store_true",
        help="Do not merge the FIQA model. Default output becomes chc_{variant}_wo_fiqa.onnx.",
    )
    parser.add_argument(
        "--disable-onnxsim",
        action="store_true",
        help="Skip automatic onnxsim simplification after saving the merged ONNX.",
    )
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help="Skip optional onnxruntime session loading after saving.",
    )
    return parser.parse_args()


def make_branch_specs(
    model_dir: Path, variant: str, include_fiqa: bool = True
) -> list[BranchSpec]:
    specs = [
        BranchSpec(
            name="bpc_is",
            model_path=model_dir / f"bpc_is_{variant}_48x48.onnx",
            source_input="images",
            public_input="head_image_48x48",
            source_output="prob_plain",
            public_output="prob_background_plain",
            batch_size=1,
        ),
        BranchSpec(
            name="mwc",
            model_path=model_dir / f"mwc_{variant}_48x48.onnx",
            source_input="images",
            public_input="head_image_48x48",
            source_output="prob_masked",
            public_output="prob_masked",
            batch_size=1,
        ),
        BranchSpec(
            name="sgc_is",
            model_path=model_dir / f"sgc_is_{variant}_48x48.onnx",
            source_input="images",
            public_input="head_image_48x48",
            source_output="prob_sunglasses",
            public_output="prob_sunglasses",
            batch_size=1,
        ),
        BranchSpec(
            name="ocec",
            model_path=model_dir / f"ocec_{variant}_24x40.onnx",
            source_input="images",
            public_input="eye_images_24x40",
            source_output="prob_open",
            public_output="prob_eye_open",
            batch_size=2,
        ),
        BranchSpec(
            name="vsdlm",
            model_path=model_dir / f"vsdlm_{variant}_30x48.onnx",
            source_input="images",
            public_input="mouth_image_30x48",
            source_output="prob_open",
            public_output="prob_mouth_open",
            batch_size=1,
        ),
    ]

    if include_fiqa:
        specs.append(
            BranchSpec(
                name="fiqa",
                model_path=model_dir / "FIQA_EdgeNeXt_XXS_1x3x352x352.onnx",
                source_input="input",
                public_input="head_image_352x352",
                source_output="quality_score",
                public_output="quality_score",
                batch_size=1,
            )
        )

    return specs


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def set_ai_onnx_opset(model: ModelProto, version: int) -> None:
    found = False
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.domain = ""
            opset.version = version
            found = True
    if not found:
        model.opset_import.append(helper.make_opsetid("", version))


def set_first_dim(value_info: ValueInfoProto, batch_size: int) -> None:
    shape = value_info.type.tensor_type.shape
    if not shape.dim:
        return
    first_dim = shape.dim[0]
    first_dim.ClearField("dim_param")
    first_dim.dim_value = batch_size


def rename_value(name: str, mapping: dict[str, str], prefix: str) -> str:
    if not name:
        return name
    return mapping.get(name, f"{prefix}{name}")


def rename_sparse_initializer(
    initializer: onnx.SparseTensorProto, mapping: dict[str, str], prefix: str
) -> None:
    if initializer.values.name:
        initializer.values.name = rename_value(initializer.values.name, mapping, prefix)
    if initializer.indices.name:
        initializer.indices.name = rename_value(initializer.indices.name, mapping, prefix)


def rewrite_branch_graph(spec: BranchSpec, model: ModelProto) -> onnx.GraphProto:
    graph = copy.deepcopy(model.graph)
    if len(graph.input) != 1:
        fail(f"{spec.model_path} must have exactly one graph input; found {len(graph.input)}")
    if len(graph.output) != 1:
        fail(f"{spec.model_path} must have exactly one graph output; found {len(graph.output)}")
    if graph.input[0].name != spec.source_input:
        fail(
            f"{spec.model_path} input is {graph.input[0].name!r}; expected {spec.source_input!r}"
        )
    if graph.output[0].name != spec.source_output:
        fail(
            f"{spec.model_path} output is {graph.output[0].name!r}; expected {spec.source_output!r}"
        )

    prefix = f"{spec.name}__"
    public_names = {
        spec.source_input: spec.public_input,
        spec.source_output: spec.public_output,
    }

    for node in graph.node:
        if node.name:
            node.name = f"{prefix}{node.name}"
        for index, input_name in enumerate(node.input):
            node.input[index] = rename_value(input_name, public_names, prefix)
        for index, output_name in enumerate(node.output):
            node.output[index] = rename_value(output_name, public_names, prefix)

    for initializer in graph.initializer:
        initializer.name = rename_value(initializer.name, public_names, prefix)
    for initializer in graph.sparse_initializer:
        rename_sparse_initializer(initializer, public_names, prefix)
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        value_info.name = rename_value(value_info.name, public_names, prefix)

    set_first_dim(graph.input[0], spec.batch_size)
    set_first_dim(graph.output[0], spec.batch_size)
    return graph


def load_source_model(spec: BranchSpec) -> ModelProto:
    if not spec.model_path.exists():
        fail(f"missing source model: {spec.model_path}")

    model = onnx.load(spec.model_path)
    if model.ir_version != TARGET_IR_VERSION:
        fail(
            f"{spec.model_path} ir_version is {model.ir_version}; expected {TARGET_IR_VERSION}"
        )

    set_ai_onnx_opset(model, TARGET_ONNX_OPSET)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        fail(f"{spec.model_path} is invalid after setting opset {TARGET_ONNX_OPSET}: {exc}")

    return model


def dedupe_inputs(existing: dict[str, ValueInfoProto], incoming: ValueInfoProto) -> None:
    current = existing.get(incoming.name)
    if current is None:
        existing[incoming.name] = incoming
        return
    if current.SerializeToString() != incoming.SerializeToString():
        fail(f"shared input {incoming.name!r} has incompatible definitions")


def build_combined_model(specs: list[BranchSpec], variant: str) -> ModelProto:
    nodes = []
    initializers = []
    sparse_initializers = []
    value_infos = []
    graph_inputs: dict[str, ValueInfoProto] = {}
    graph_outputs = []

    for spec in specs:
        model = load_source_model(spec)
        graph = rewrite_branch_graph(spec, model)
        nodes.extend(graph.node)
        initializers.extend(graph.initializer)
        sparse_initializers.extend(graph.sparse_initializer)
        value_infos.extend(graph.value_info)
        for graph_input in graph.input:
            dedupe_inputs(graph_inputs, graph_input)
        graph_outputs.extend(graph.output)

    graph = helper.make_graph(
        nodes=nodes,
        name=f"chc_{variant}",
        inputs=list(graph_inputs.values()),
        outputs=graph_outputs,
        initializer=initializers,
        sparse_initializer=sparse_initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", TARGET_ONNX_OPSET)],
        producer_name="build_chc_onnx.py",
    )
    model.ir_version = TARGET_IR_VERSION
    onnx.checker.check_model(model)
    return model


def runtime_check(output_path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed; skipped runtime load check")
        return

    ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    print("onnxruntime load check: ok")


def simplify_model(output_path: Path) -> None:
    try:
        from onnxsim import simplify
    except ImportError:
        fail("onnxsim is not installed; install it or pass --disable-onnxsim")

    simplified_model, check_ok = simplify(str(output_path))
    if not check_ok:
        fail(f"onnxsim validation failed for {output_path}")

    onnx.checker.check_model(simplified_model)
    onnx.save(simplified_model, output_path)
    print(f"onnxsim simplified: {output_path}")


def make_verify_inputs() -> dict[str, object]:
    import numpy as np

    rng = np.random.default_rng(0)
    return {
        "head_image_48x48": rng.standard_normal((1, 3, 48, 48), dtype=np.float32),
        "eye_images_24x40": rng.standard_normal((2, 3, 24, 40), dtype=np.float32),
        "mouth_image_30x48": rng.standard_normal((1, 3, 30, 48), dtype=np.float32),
        "head_image_352x352": rng.standard_normal((1, 3, 352, 352), dtype=np.float32),
    }


def verify_outputs(specs: list[BranchSpec], output_path: Path) -> None:
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        fail("--verify requires onnxruntime")

    inputs = make_verify_inputs()
    merged_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    merged_inputs = {
        session_input.name: inputs[session_input.name]
        for session_input in merged_session.get_inputs()
    }
    merged_outputs = dict(
        zip(
            [output.name for output in merged_session.get_outputs()],
            merged_session.run(None, merged_inputs),
        )
    )

    for spec in specs:
        source_session = ort.InferenceSession(
            str(spec.model_path), providers=["CPUExecutionProvider"]
        )
        source_output = source_session.run(
            None, {spec.source_input: inputs[spec.public_input]}
        )[0]
        merged_output = merged_outputs[spec.public_output]
        max_abs_diff = float(np.max(np.abs(source_output - merged_output)))
        if max_abs_diff != 0.0:
            fail(
                f"{spec.public_output} differs from {spec.model_path}: "
                f"max_abs_diff={max_abs_diff}"
            )
        print(f"verify {spec.public_output}: max_abs_diff=0.0")


def main() -> int:
    args = parse_args()
    variant = args.variant.strip()
    if not variant:
        fail("--variant must not be empty")

    model_dir = args.model_dir.resolve()
    default_output = f"chc_{variant}{'_wo_fiqa' if args.disable_fiqa else ''}.onnx"
    output_path = (args.output or Path(default_output)).resolve()
    specs = make_branch_specs(model_dir, variant, include_fiqa=not args.disable_fiqa)

    missing = [str(spec.model_path) for spec in specs if not spec.model_path.exists()]
    if missing:
        fail("missing source models:\n  " + "\n  ".join(missing))

    combined_model = build_combined_model(specs, variant)
    onnx.save(combined_model, output_path)
    print(f"saved: {output_path}")

    if not args.disable_onnxsim:
        simplify_model(output_path)
    if not args.skip_runtime_check:
        runtime_check(output_path)
    if args.verify:
        verify_outputs(specs, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
