"""Unit tests for host_spec helper."""

from __future__ import annotations

from airllm_bench.services.host_spec import capture_host_spec, write_host_spec


class TestCaptureHostSpec:
    def test_required_keys_present(self):
        spec = capture_host_spec()
        required = {
            "cpu", "cpu_count_logical", "cpu_count_physical",
            "total_ram_mb", "free_ram_mb", "os", "python", "has_cuda",
        }
        assert required.issubset(spec.keys())

    def test_has_cuda_is_bool(self):
        spec = capture_host_spec()
        assert isinstance(spec["has_cuda"], bool)

    def test_total_ram_positive(self):
        spec = capture_host_spec()
        assert spec["total_ram_mb"] > 0

    def test_cpu_count_logical_gte_physical(self):
        spec = capture_host_spec()
        assert spec["cpu_count_logical"] >= spec["cpu_count_physical"]

    def test_no_cuda_on_cpu_only_host(self):
        """On this CPU-only host, has_cuda must be False."""
        spec = capture_host_spec()
        assert spec["has_cuda"] is False


class TestWriteHostSpec:
    def test_writes_json_file(self, tmp_path):
        import json

        path = write_host_spec(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "cpu" in data

    def test_creates_dir_if_absent(self, tmp_path):
        new_dir = tmp_path / "new_results"
        assert not new_dir.exists()
        write_host_spec(new_dir)
        assert new_dir.exists()
