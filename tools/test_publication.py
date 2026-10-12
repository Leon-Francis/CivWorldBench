"""Publication metadata tests use only synthetic identities and no network."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import configure_publication as publication


class PublicationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = {
            "title": "Example Benchmark: A Research Report", "project": "Example Benchmark",
            "version": "1.0.0", "date": "2026-10-12",
            "authors": [{"name": "Example Author", "given_names": "Example", "family_names": "Author",
                         "affiliations": [1], "corresponding": True}],
            "affiliations": [{"id": 1, "name": "Example University"}],
            "contact": {"name": "Example Author", "email": "author@example.org"},
            "github": {"repo_id": "example/benchmark", "pages_url": "https://example.org/benchmark/"},
            "dataset": {"status": "in_preparation", "repo_id": None, "revision": None},
            "paper": {"arxiv_id": None, "url": "https://example.org/benchmark/paper.pdf",
                      "pdf_url": "https://example.org/benchmark/paper.pdf"},
        }
        self.write_config()

    def write_config(self):
        (self.root / "PUBLICATION.json").write_text(json.dumps(self.config))

    def run_silently(self, arguments=()):
        with redirect_stdout(io.StringIO()), patch("socket.socket", side_effect=AssertionError("network")):
            return publication.main(arguments, root=self.root)

    def test_generate_and_check_without_invented_publication_claims(self):
        original = (self.root / "PUBLICATION.json").read_bytes()
        self.assertEqual(self.run_silently(), 0)
        self.assertEqual(self.run_silently(["--check"]), 0)
        self.assertEqual((self.root / "PUBLICATION.json").read_bytes(), original)
        cff = json.loads((self.root / "CITATION.cff").read_text())
        self.assertEqual(cff["authors"][0]["family-names"], "Author")
        self.assertEqual(cff["contact"][0]["email"], "author@example.org")
        self.assertEqual(cff["preferred-citation"]["url"], self.config["paper"]["url"])
        bib = (self.root / "CITATION.bib").read_text()
        self.assertIn("Author, Example", bib)
        for field in ("journal", "booktitle", "eprint", "archivePrefix", "doi"):
            self.assertNotIn(field + " =", bib)

    def test_check_detects_stale_files_without_rewriting(self):
        self.run_silently()
        path = self.root / "CITATION.bib"
        path.write_text("outdated\n")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            self.run_silently(["--check"])
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(path.read_text(), "outdated\n")

    def test_dataset_repository_and_arxiv_claims_are_rejected(self):
        for section, field, value in (
            ("dataset", "repo_id", "example/benchmark"),
            ("dataset", "revision", "main"),
            ("dataset", "status", "published"),
            ("paper", "arxiv_id", "not-confirmed"),
        ):
            config = deepcopy(self.config)
            config[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                publication.validate(config)

    def test_invalid_urls_and_affiliation_references_are_rejected(self):
        for value in ("http://example.org/benchmark/", "https://127.0.0.1/benchmark/",
                      "https://example.org/with space/", "https://example.org/benchmark/?token=example"):
            config = deepcopy(self.config)
            config["github"]["pages_url"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                publication.validate(config)
        config = deepcopy(self.config)
        config["authors"][0]["affiliations"] = [2]
        with self.assertRaises(ValueError):
            publication.validate(config)

    def test_stale_dataset_link_in_website_is_rejected(self):
        self.run_silently()
        directory = self.root / "website/src"
        directory.mkdir(parents=True)
        (directory / "links.ts").write_text('const data = "https://huggingface.co/datasets/example/benchmark";')
        with redirect_stderr(io.StringIO()) as output, self.assertRaises(SystemExit) as raised:
            self.run_silently(["--check"])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("website/src/links.ts", output.getvalue())

    def test_symlinked_citation_is_not_overwritten(self):
        target = self.root / "original.txt"
        target.write_text("preserve\n")
        (self.root / "CITATION.cff").symlink_to(target)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_silently()
        self.assertEqual(target.read_text(), "preserve\n")


if __name__ == "__main__":
    unittest.main()
