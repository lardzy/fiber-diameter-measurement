from __future__ import annotations

from pathlib import Path
import unittest


class DockerContractTests(unittest.TestCase):
    def test_image_runs_non_root_and_is_fail_closed_by_default(self) -> None:
        dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
        content = dockerfile.read_text(encoding="utf-8")

        self.assertIn("USER 10001:10001", content)
        self.assertIn("AREA_ALLOW_ANONYMOUS_DEV=0", content)
        self.assertIn("AREA_REQUIRE_TRUSTED_WEIGHTS=1", content)
        self.assertIn("AREA_VERIFY_TRUSTED_WEIGHTS=1", content)
        self.assertIn("AREA_MAX_CACHED_MODELS=2", content)
        self.assertIn("http://127.0.0.1:9001/ready", content)
        self.assertIn('"--limit-concurrency", "4"', content)
        self.assertIn("AREA_MAX_MASK_WORKING_BYTES=1610612736", content)

        main_source = (dockerfile.parent / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn('@app.get("/live")', main_source)
        self.assertIn('@app.get("/ready")', main_source)
        self.assertIn('@app.post("/v1/infer", dependencies=[Depends(require_api_auth)])', main_source)


if __name__ == "__main__":
    unittest.main()
