from __future__ import annotations

from contextlib import nullcontext
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sageattention_validate_wheel_runtime",
    ROOT / "scripts" / "validate-wheel.py",
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


class FakeDType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"torch.{self.name}"


FLOAT16 = FakeDType("float16")
BFLOAT16 = FakeDType("bfloat16")
FLOAT32 = FakeDType("float32")


class FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value

    def __mul__(self, other: FakeScalar) -> FakeScalar:
        return FakeScalar(self.value * other.value)

    def __truediv__(self, other: FakeScalar) -> FakeScalar:
        return FakeScalar(self.value / other.value)

    def clamp_min(self, minimum: float) -> FakeScalar:
        return FakeScalar(max(self.value, minimum))


class FakeTensor:
    def __init__(
        self,
        values: list[float],
        *,
        shape: tuple[int, ...],
        dtype: FakeDType = FLOAT16,
        is_cuda: bool = True,
    ) -> None:
        self.values = values
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = is_cuda

    def float(self) -> FakeTensor:
        return FakeTensor(
            list(self.values),
            shape=self.shape,
            dtype=FLOAT32,
            is_cuda=self.is_cuda,
        )

    def flatten(self) -> FakeTensor:
        return FakeTensor(
            list(self.values),
            shape=(len(self.values),),
            dtype=self.dtype,
            is_cuda=self.is_cuda,
        )

    def __sub__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(
            [left - right for left, right in zip(self.values, other.values)],
            shape=self.shape,
            dtype=self.dtype,
            is_cuda=self.is_cuda,
        )


class FakeFiniteResult:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> bool:
        return self.value


def fake_torch_module(shape: tuple[int, ...]) -> ModuleType:
    torch = ModuleType("torch")
    torch.__path__ = []  # type: ignore[attr-defined]
    torch.__version__ = "2.10.0+cu128"  # type: ignore[attr-defined]
    torch.Tensor = FakeTensor  # type: ignore[attr-defined]
    torch.float16 = FLOAT16  # type: ignore[attr-defined]
    torch.bfloat16 = BFLOAT16  # type: ignore[attr-defined]
    torch.version = SimpleNamespace(cuda="12.8")  # type: ignore[attr-defined]
    torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        get_device_capability=lambda _index: (9, 0),
        get_device_name=lambda _index: "Fake H100",
        is_available=lambda: True,
        synchronize=lambda: None,
    )
    torch.linalg = SimpleNamespace(  # type: ignore[attr-defined]
        vector_norm=lambda tensor: FakeScalar(
            math.sqrt(sum(value * value for value in tensor.values))
        )
    )
    torch.dot = lambda left, right: FakeScalar(  # type: ignore[attr-defined]
        sum(a * b for a, b in zip(left.values, right.values))
    )
    torch.isfinite = lambda tensor: FakeFiniteResult(  # type: ignore[attr-defined]
        all(math.isfinite(value) for value in tensor.values)
    )
    torch.manual_seed = lambda _seed: None  # type: ignore[attr-defined]
    torch.randn = lambda *_args, **_kwargs: FakeTensor(  # type: ignore[attr-defined]
        [1.0, 0.0], shape=shape
    )
    return torch


def runtime_fixture(*, inject_failures: bool) -> tuple[dict, dict, dict, dict, list]:
    shape = (1, 8, 512, 64)
    torch = fake_torch_module(shape)
    functional = ModuleType("torch.nn.functional")
    functional.scaled_dot_product_attention = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: FakeTensor([1.0, 0.0], shape=shape, dtype=FLOAT32)
    )
    attention = ModuleType("torch.nn.attention")
    attention.SDPBackend = SimpleNamespace(MATH="math")  # type: ignore[attr-defined]
    attention.sdpa_kernel = lambda _backend: nullcontext()  # type: ignore[attr-defined]
    nn = ModuleType("torch.nn")
    nn.__path__ = []  # type: ignore[attr-defined]
    nn.functional = functional  # type: ignore[attr-defined]
    nn.attention = attention  # type: ignore[attr-defined]
    torch.nn = nn  # type: ignore[attr-defined]

    calls: list[tuple[str, bool]] = []

    def implementation(name: str):
        def invoke(*_args, **kwargs):
            causal = kwargs["is_causal"]
            calls.append((name, causal))
            if inject_failures and name == "sageattn_dispatch" and not causal:
                return FakeTensor([0.0, 1.0], shape=shape)
            if inject_failures and name == "qattn_sm90_cuda" and not causal:
                raise RuntimeError("simulated kernel launch failure")
            return FakeTensor([1.0, 0.0], shape=shape)

        return invoke

    sageattention = ModuleType("sageattention")
    sageattention.sageattn = implementation("sageattn_dispatch")  # type: ignore[attr-defined]
    sageattention.sageattn_qk_int8_pv_fp16_cuda = implementation(  # type: ignore[attr-defined]
        "qattn_sm80_cuda"
    )
    sageattention.sageattn_qk_int8_pv_fp8_cuda = implementation(  # type: ignore[attr-defined]
        "qattn_sm89_cuda"
    )
    sageattention.sageattn_qk_int8_pv_fp8_cuda_sm90 = implementation(  # type: ignore[attr-defined]
        "qattn_sm90_cuda"
    )

    modules: dict[str, ModuleType] = {
        "torch": torch,
        "torch.nn": nn,
        "torch.nn.functional": functional,
        "torch.nn.attention": attention,
        "sageattention": sageattention,
    }
    for name in (
        "sageattention._fused",
        "sageattention._qattn_sm80",
        "sageattention._qattn_sm89",
        "sageattention._qattn_sm90",
    ):
        module = ModuleType(name)
        module.__file__ = f"/fake/{name.rsplit('.', 1)[-1]}.so"
        modules[name] = module

    policy = {
        "reference": "torch.nn.functional.scaled_dot_product_attention",
        "minimum_cosine_similarity": 0.995,
        "maximum_relative_l2": 0.1,
        "implementations_by_capability": {
            "9.0": [
                "sageattn_dispatch",
                "qattn_sm80_cuda",
                "qattn_sm89_cuda",
                "qattn_sm90_cuda",
            ]
        },
        "canonical_case": {
            "seed": 2026,
            "batch_size": shape[0],
            "query_heads": shape[1],
            "key_value_heads": shape[1],
            "sequence_length": shape[2],
            "head_dimension": shape[3],
            "dtype": "float16",
            "tensor_layout": "HND",
            "causal_modes": [False, True],
        },
    }
    matrix = {
        "validation": {
            "representative_gpu_capabilities": [
                {"compute_capability": "9.0", "required": True}
            ],
            "runtime_numeric": policy,
        }
    }
    build = {
        "id": "cp312-torch2.10.0-cu128",
        "torch_version": "2.10.0+cu128",
        "torch_cuda_version": "12.8",
        "wheel_version": "2.2.0",
        "comfyui_runtime_image": "example/runtime@sha256:" + "1" * 64,
    }
    artifact = {
        "asset": "sageattention.whl",
        "sha256": "2" * 64,
    }
    return modules, matrix, build, artifact, calls


class RuntimeValidationReportTests(unittest.TestCase):
    def run_fixture(self, *, inject_failures: bool, report_path: Path):
        modules, matrix, build, artifact, calls = runtime_fixture(
            inject_failures=inject_failures
        )
        environment = {
            "RUNTIME_IMAGE_REF": build["comfyui_runtime_image"],
        }
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(importlib.metadata, "version", return_value="2.2.0"),
        ):
            report = VALIDATOR.run_runtime_validation(
                matrix,
                build,
                artifact,
                "9.0",
                report_path,
            )
        return report, calls

    def test_failure_report_collects_metrics_errors_and_later_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "sm90.json"
            report_path.write_text('{"status":"stale"}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError,
                r"failed for 2 of 8 outcomes; diagnostic report:.*sm90\.json",
            ):
                self.run_fixture(inject_failures=True, report_path=report_path)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "fail")
            self.assertEqual(len(report["results"]), 8)
            self.assertEqual(len(report["failures"]), 2)
            self.assertFalse(report_path.with_name(".sm90.json.tmp").exists())
            self.assertEqual(
                {
                    (result["implementation"], result["causal"])
                    for result in report["results"]
                },
                {
                    (implementation, causal)
                    for implementation in (
                        "sageattn_dispatch",
                        "qattn_sm80_cuda",
                        "qattn_sm89_cuda",
                        "qattn_sm90_cuda",
                    )
                    for causal in (False, True)
                },
            )

            numeric_failure = next(
                result
                for result in report["results"]
                if result["implementation"] == "sageattn_dispatch"
                and result["causal"] is False
            )
            self.assertEqual(numeric_failure["cosine_similarity"], 0.0)
            self.assertAlmostEqual(numeric_failure["relative_l2"], math.sqrt(2.0))
            self.assertEqual(
                {error["stage"] for error in numeric_failure["errors"]},
                {"numeric-validation"},
            )

            execution_failure = next(
                result
                for result in report["results"]
                if result["implementation"] == "qattn_sm90_cuda"
                and result["causal"] is False
            )
            self.assertEqual(
                execution_failure["errors"],
                [{
                    "message": "simulated kernel launch failure",
                    "stage": "execution",
                    "type": "RuntimeError",
                }],
            )
            final_result = report["results"][-1]
            self.assertEqual(
                (final_result["implementation"], final_result["causal"]),
                ("qattn_sm90_cuda", True),
            )
            self.assertNotIn("errors", final_result)

    def test_pass_report_retains_strict_complete_result_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "sm90.json"
            report, calls = self.run_fixture(
                inject_failures=False,
                report_path=report_path,
            )

            self.assertEqual(report["status"], "pass")
            self.assertNotIn("failures", report)
            self.assertEqual(len(report["results"]), 8)
            self.assertEqual(len(calls), 8)
            for result in report["results"]:
                self.assertNotIn("errors", result)
                self.assertEqual(result["output_shape"], [1, 8, 512, 64])
                self.assertEqual(result["output_dtype"], "float16")
                self.assertGreaterEqual(result["cosine_similarity"], 0.995)
                self.assertLessEqual(result["relative_l2"], 0.1)
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8")),
                report,
            )

    def test_output_type_shape_dtype_and_finite_checks_are_preserved(self) -> None:
        shape = [1, 8, 512, 64]
        torch = fake_torch_module(tuple(shape))
        bad_dtype = FakeDType("float64")
        result = VALIDATOR._evaluate_runtime_case(
            torch_module=torch,
            implementation=lambda _causal: FakeTensor(
                [math.nan, 0.0],
                shape=(1, 1, 1, 2),
                dtype=bad_dtype,
                is_cuda=False,
            ),
            implementation_name="qattn_sm90_cuda",
            causal=False,
            reference=FakeTensor([1.0, 0.0], shape=tuple(shape), dtype=FLOAT32),
            expected_output_shape=shape,
            expected_output_dtype="float16",
            expected_dtype=FLOAT16,
            policy={
                "minimum_cosine_similarity": 0.995,
                "maximum_relative_l2": 0.1,
            },
        )

        messages = [error["message"] for error in result["errors"]]
        self.assertTrue(any("non-CUDA output" in message for message in messages))
        self.assertTrue(any("output shape" in message for message in messages))
        self.assertTrue(any("output dtype" in message for message in messages))
        self.assertTrue(any("non-finite output" in message for message in messages))
        self.assertNotIn("cosine_similarity", result)
        self.assertNotIn("relative_l2", result)


if __name__ == "__main__":
    unittest.main()
