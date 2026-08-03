"""Config, run state / resume, hardware resolution, and crawler integration."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from autotrain.crawler.crawler import crawl_sync
from autotrain.crawler.sitemap import parse_sitemap_bytes
from autotrain.trainer.hardware import Hardware, resolve_training_setup
from autotrain.utils.config import apply_overrides, load_config
from autotrain.utils.paths import Workspace, slugify_domain
from autotrain.utils.state import PHASES, StateStore


class TestConfig:
    def test_defaults_load(self) -> None:
        cfg = load_config(domain="https://e.com")
        assert cfg.domain == "https://e.com"
        assert cfg.crawler.max_pages > 0
        assert len(cfg.dataset.sample_types) == 8

    def test_override_types_coerced(self) -> None:
        cfg = load_config(
            domain="https://e.com",
            overrides=[
                "crawler.max_pages=42",
                "crawler.respect_robots=false",
                "training.learning_rate=1e-5",
                "runtime.log_level=DEBUG",
            ],
        )
        assert cfg.crawler.max_pages == 42
        assert cfg.crawler.respect_robots is False
        assert cfg.training.learning_rate == pytest.approx(1e-5)
        assert cfg.runtime.log_level == "DEBUG"

    def test_list_override(self) -> None:
        cfg = load_config(domain="https://e.com", overrides=["dataset.sample_types=doc_qa,explanation"])
        assert cfg.dataset.sample_types == ["doc_qa", "explanation"]

    def test_seed_propagates(self) -> None:
        cfg = load_config(domain="https://e.com", overrides=["runtime.seed=7"])
        assert cfg.split.seed == 7 and cfg.training.seed == 7

    def test_invalid_regex_rejected(self) -> None:
        with pytest.raises(Exception):
            load_config(domain="https://e.com", overrides=["crawler.include_patterns=[unclosed"])

    def test_invalid_ratio_rejected(self) -> None:
        with pytest.raises(Exception):
            load_config(domain="https://e.com", overrides=["split.train_ratio=1.5"])

    def test_malformed_override_rejected(self) -> None:
        with pytest.raises(ValueError):
            apply_overrides({}, ["no_equals_sign"])


class TestSlug:
    @pytest.mark.parametrize(
        "domain,expected",
        [
            ("https://example.com", "example.com"),
            ("example.com", "example.com"),
            ("https://docs.example.co.uk/guide/", "docs.example.co.uk_guide"),
            ("https://example.com/", "example.com"),
        ],
    )
    def test_slugify(self, domain: str, expected: str) -> None:
        assert slugify_domain(domain) == expected


class TestState:
    def test_roundtrip(self, tmp_path) -> None:
        store = StateStore(tmp_path / "run.json")
        store.state.domain = "https://e.com"
        store.start("discover")
        store.complete("discover", {"pages": 10})

        reloaded = StateStore(tmp_path / "run.json")
        assert reloaded.state.domain == "https://e.com"
        assert reloaded.state.phases["discover"].status == "completed"
        assert reloaded.state.phases["discover"].summary["pages"] == 10

    def test_should_run_when_pending(self, tmp_path) -> None:
        store = StateStore(tmp_path / "run.json")
        assert store.should_run("discover", [], resume=True)

    def test_skips_completed_with_outputs(self, tmp_path) -> None:
        artifact = tmp_path / "out.json"
        artifact.write_text('{"a": 1}', encoding="utf-8")
        store = StateStore(tmp_path / "run.json")
        store.start("discover")
        store.complete("discover")
        assert not store.should_run("discover", [artifact], resume=True)

    def test_missing_output_rearms_phase(self, tmp_path) -> None:
        artifact = tmp_path / "out.json"
        artifact.write_text("{}", encoding="utf-8")
        store = StateStore(tmp_path / "run.json")
        store.start("discover")
        store.complete("discover")
        artifact.unlink()
        assert store.should_run("discover", [artifact], resume=True)

    def test_empty_output_rearms_phase(self, tmp_path) -> None:
        artifact = tmp_path / "out.json"
        artifact.write_text("", encoding="utf-8")
        store = StateStore(tmp_path / "run.json")
        store.start("discover")
        store.complete("discover")
        assert store.should_run("discover", [artifact], resume=True)

    def test_force_overrides_resume(self, tmp_path) -> None:
        artifact = tmp_path / "out.json"
        artifact.write_text("{}", encoding="utf-8")
        store = StateStore(tmp_path / "run.json")
        store.start("discover")
        store.complete("discover")
        assert store.should_run("discover", [artifact], resume=True, force=True)

    def test_corrupt_state_recovers(self, tmp_path) -> None:
        path = tmp_path / "run.json"
        path.write_text("{{{ not json", encoding="utf-8")
        store = StateStore(path)  # must not raise
        assert store.state.phases == {}

    def test_all_phases_known(self) -> None:
        assert len(PHASES) == 12


class TestHardwareResolution:
    @pytest.fixture
    def training_cfg(self):
        return load_config(domain="https://e.com").training

    def test_cpu_disables_4bit_and_mixed_precision(self, training_cfg) -> None:
        resolved = resolve_training_setup(training_cfg, Hardware(device="cpu"))
        assert resolved.load_in_4bit is False
        assert resolved.bf16 is False and resolved.fp16 is False
        assert resolved.torch_dtype == "float32"
        assert resolved.optim == "adamw_torch"
        assert any("4-bit" in d for d in resolved.downgrades)

    def test_cpu_picks_small_model(self, training_cfg) -> None:
        resolved = resolve_training_setup(training_cfg, Hardware(device="cpu"))
        assert resolved.model_id == training_cfg.cpu_model

    def test_cuda_enables_qlora(self, training_cfg) -> None:
        hardware = Hardware(
            device="cuda", device_name="A100", vram_gb=80,
            compute_capability=(8, 0), supports_bf16=True,
            has_bitsandbytes=True, has_flash_attention=True,
        )
        resolved = resolve_training_setup(training_cfg, hardware)
        assert resolved.load_in_4bit is True
        assert resolved.bf16 is True
        assert resolved.optim == "paged_adamw_8bit"
        assert resolved.attn_implementation == "flash_attention_2"
        assert resolved.model_id == training_cfg.cuda_model
        assert resolved.downgrades == []

    def test_cuda_without_bitsandbytes_downgrades(self, training_cfg) -> None:
        hardware = Hardware(
            device="cuda", compute_capability=(8, 0),
            supports_bf16=True, has_bitsandbytes=False,
        )
        resolved = resolve_training_setup(training_cfg, hardware)
        assert resolved.load_in_4bit is False
        assert any("bitsandbytes" in d for d in resolved.downgrades)

    def test_pre_ampere_uses_fp16(self, training_cfg) -> None:
        hardware = Hardware(
            device="cuda", device_name="T4", compute_capability=(7, 5),
            supports_bf16=False, has_bitsandbytes=True,
        )
        resolved = resolve_training_setup(training_cfg, hardware)
        assert resolved.bf16 is False and resolved.fp16 is True

    def test_explicit_cuda_without_gpu_raises(self, training_cfg) -> None:
        training_cfg.device = "cuda"
        with pytest.raises(RuntimeError, match="no CUDA device"):
            resolve_training_setup(training_cfg, Hardware(device="cpu"))


class TestSitemapParsing:
    def test_urlset(self) -> None:
        xml = b"""<?xml version="1.0"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://e.com/a</loc><lastmod>2024-01-01</lastmod></url>
          <url><loc>https://e.com/b</loc></url>
        </urlset>"""
        entries, nested = parse_sitemap_bytes(xml, "https://e.com/sitemap.xml")
        assert [e.loc for e in entries] == ["https://e.com/a", "https://e.com/b"]
        assert entries[0].lastmod == "2024-01-01"
        assert nested == []

    def test_sitemap_index(self) -> None:
        xml = b"""<?xml version="1.0"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <sitemap><loc>https://e.com/s1.xml</loc></sitemap>
          <sitemap><loc>https://e.com/s2.xml</loc></sitemap>
        </sitemapindex>"""
        entries, nested = parse_sitemap_bytes(xml, "https://e.com/sitemap.xml")
        assert entries == []
        assert nested == ["https://e.com/s1.xml", "https://e.com/s2.xml"]

    def test_gzipped(self) -> None:
        import gzip

        xml = b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://e.com/a</loc></url></urlset>'
        entries, _ = parse_sitemap_bytes(gzip.compress(xml), "https://e.com/sitemap.xml.gz")
        assert len(entries) == 1

    def test_malformed_tail_recovered(self) -> None:
        xml = b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://e.com/a</loc></url><url><loc>trunc'
        entries, _ = parse_sitemap_bytes(xml, "https://e.com/sitemap.xml")
        assert len(entries) >= 1

    def test_garbage_returns_empty(self) -> None:
        entries, nested = parse_sitemap_bytes(b"not xml at all", "https://e.com/s.xml")
        assert entries == [] and nested == []


PAGE = """<!doctype html><html><head><title>{title}</title></head><body>
<nav><a href="/">Home</a><a href="/a">A</a><a href="/b">B</a></nav>
<main><h1>{title}</h1><p>{body}</p></main>
<footer><a href="https://external.example.org/x">External</a></footer>
</body></html>"""


class TestCrawlerIntegration:
    @respx.mock
    def test_crawls_links_and_respects_scope(self, tmp_path) -> None:
        respx.get("https://site.test/robots.txt").mock(return_value=httpx.Response(404))
        for path in ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml",
                     "/sitemap.xml.gz", "/sitemaps/sitemap.xml", "/wp-sitemap.xml"):
            respx.get(f"https://site.test{path}").mock(return_value=httpx.Response(404))

        for path, title in (("/", "Home"), ("/a", "Page A"), ("/b", "Page B")):
            respx.get(f"https://site.test{path}").mock(
                return_value=httpx.Response(
                    200,
                    html=PAGE.format(title=title, body="Body text. " * 30),
                    headers={"content-type": "text/html; charset=utf-8"},
                )
            )
        respx.get("https://external.example.org/x").mock(return_value=httpx.Response(200, html="<p>no</p>"))

        cfg = load_config(
            domain="https://site.test",
            overrides=[f"runtime.workspace={tmp_path.as_posix()}", "crawler.concurrency=2"],
        )
        ws = Workspace(tmp_path / "site.test").ensure()
        report = crawl_sync(cfg, ws)

        assert report["pages"]["stored_ok"] == 3
        assert report["skipped"]["offsite"] >= 1
        assert ws.urls_json.exists() and ws.crawl_report.exists() and ws.sitemap_json.exists()
        assert len(list(ws.raw_html.glob("page_*.html"))) == 3

    @respx.mock
    def test_robots_disallow_is_honoured(self, tmp_path) -> None:
        respx.get("https://site.test/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        )
        for path in ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml",
                     "/sitemap.xml.gz", "/sitemaps/sitemap.xml", "/wp-sitemap.xml"):
            respx.get(f"https://site.test{path}").mock(return_value=httpx.Response(404))

        home = ('<html><body><main><p>hello there this is content</p>'
                '<a href="/private/secret">secret</a><a href="/ok">ok</a></main></body></html>')
        respx.get("https://site.test/").mock(
            return_value=httpx.Response(200, html=home, headers={"content-type": "text/html"})
        )
        respx.get("https://site.test/ok").mock(
            return_value=httpx.Response(200, html="<html><body><p>allowed page</p></body></html>",
                                        headers={"content-type": "text/html"})
        )
        secret = respx.get("https://site.test/private/secret").mock(
            return_value=httpx.Response(200, html="<p>secret</p>")
        )

        cfg = load_config(
            domain="https://site.test",
            overrides=[f"runtime.workspace={tmp_path.as_posix()}", "crawler.concurrency=1"],
        )
        ws = Workspace(tmp_path / "site.test").ensure()
        report = crawl_sync(cfg, ws)

        assert not secret.called, "crawler fetched a robots.txt-disallowed path"
        assert report["skipped"]["robots"] >= 1

    @respx.mock
    def test_duplicate_bodies_stored_once(self, tmp_path) -> None:
        respx.get("https://site.test/robots.txt").mock(return_value=httpx.Response(404))
        for path in ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml",
                     "/sitemap.xml.gz", "/sitemaps/sitemap.xml", "/wp-sitemap.xml"):
            respx.get(f"https://site.test{path}").mock(return_value=httpx.Response(404))

        body = ('<html><body><main><p>identical content everywhere</p>'
                '<a href="/dup1">1</a><a href="/dup2">2</a></main></body></html>')
        for path in ("/", "/dup1", "/dup2"):
            respx.get(f"https://site.test{path}").mock(
                return_value=httpx.Response(200, html=body, headers={"content-type": "text/html"})
            )

        cfg = load_config(
            domain="https://site.test",
            overrides=[f"runtime.workspace={tmp_path.as_posix()}", "crawler.concurrency=1"],
        )
        ws = Workspace(tmp_path / "site.test").ensure()
        report = crawl_sync(cfg, ws)

        assert report["skipped"]["duplicate_content"] >= 2
        assert len(list(ws.raw_html.glob("page_*.html"))) == 1
