# Publication Metadata

The public resources are the
[Project Page](https://leon-francis.github.io/CivWorldBench/) and
[Paper](https://leon-francis.github.io/CivWorldBench/CivWorldBench.pdf).

Dataset: **In preparation**. This code release has no dataset repository,
revision, data archives, or download endpoint. Public metadata must retain this
status until a separate data release is authorized and completed.

`PUBLICATION.json` records approved public authorship and resource locations.
Credentials, private host paths, and access tokens are never publication
metadata. Code, the paper, and approved Project Page presentation assets are the
public scope of this release.

Generate and verify the citation files after reviewing `PUBLICATION.json`:

```sh
python tools/configure_publication.py
python tools/configure_publication.py --check
```

The tool writes only `CITATION.cff` and `CITATION.bib`. The check command does not
write files: it validates the confirmed authors and affiliations, public HTTPS
resource URLs, citation consistency, and the absence of Hugging Face dataset
links in public metadata, documentation, and website text. The current release
requires `paper.arxiv_id` to remain null and does not invent a venue or DOI.

For the code checks:

```sh
python -m unittest discover -s tools -p 'test_*.py'
```
