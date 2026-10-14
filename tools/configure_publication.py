"""Generate public citation files from confirmed publication metadata."""
from __future__ import annotations

import argparse
from datetime import date
from email.headerregistry import Address
import html
import ipaddress
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATASET_LINK = re.compile(
    r"(?i)(?:https?:)?//(?:www\.)?huggingface\.co/(?:datasets|api/datasets)(?:[/?#\s\"'<>)]|$)"
)
TEXT_SUFFIXES = {".md", ".txt", ".json", ".html", ".js", ".ts", ".tsx", ".jsx", ".vue", ".svelte", ".cff", ".bib"}


def require_keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) <= set(required) | set(optional):
        raise ValueError("Publication metadata has missing or unsupported fields")


def text(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError(f"Invalid {field}")
    return value


def public_url(value, field):
    value = text(value, field)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError(f"Invalid {field}") from None
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or port is not None
            or not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?", host)
            or "." not in host or ".." in host or "%" in value or "\\" in value
            or any(character.isspace() for character in value)
            or any(not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                   for label in host.split("."))
            or any(part in {".", ".."} for part in parsed.path.split("/"))):
        raise ValueError(f"Invalid public HTTPS URL in {field}")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError(f"Public URLs must use a named host in {field}")
    if host.lower().endswith((".local", ".localhost", ".internal")):
        raise ValueError(f"Invalid public host in {field}")
    return value


def validate(config):
    require_keys(config, {"title", "project", "version", "date", "authors", "affiliations",
                          "contact", "github", "dataset", "paper"})
    for key in ("title", "project", "version"):
        text(config[key], key)
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?", config["version"]):
        raise ValueError("Invalid publication version")
    try:
        parsed_date = date.fromisoformat(config["date"])
    except (TypeError, ValueError):
        raise ValueError("Invalid publication date") from None
    if parsed_date.isoformat() != config["date"]:
        raise ValueError("Publication date must use YYYY-MM-DD")

    if not isinstance(config["affiliations"], list) or not config["affiliations"]:
        raise ValueError("Affiliations must be a nonempty list")
    institutions = {}
    for item in config["affiliations"]:
        require_keys(item, {"id", "name"})
        number = item["id"]
        if type(number) is not int or number < 1 or number in institutions:
            raise ValueError("Affiliation IDs must be unique positive integers")
        institutions[number] = text(item["name"], "affiliation name")
    if not isinstance(config["authors"], list) or not config["authors"]:
        raise ValueError("Authors must be a nonempty ordered list")
    names = set()
    corresponding = 0
    for author in config["authors"]:
        require_keys(author, {"name", "given_names", "family_names", "affiliations"}, {"corresponding"})
        for field in ("name", "given_names", "family_names"):
            text(author[field], "author " + field)
        if author["name"] != author["given_names"] + " " + author["family_names"] or author["name"] in names:
            raise ValueError("Author names must be unique and match their given and family names")
        names.add(author["name"])
        identifiers = author["affiliations"]
        if (not isinstance(identifiers, list) or not identifiers
                or any(type(i) is not int or i not in institutions for i in identifiers)
                or len(set(identifiers)) != len(identifiers)):
            raise ValueError("Author affiliation references are invalid")
        if "corresponding" in author and type(author["corresponding"]) is not bool:
            raise ValueError("Corresponding-author flags must be booleans")
        corresponding += author.get("corresponding", False)
    if corresponding > 1:
        raise ValueError("This release supports one corresponding author")

    require_keys(config["contact"], {"name", "email"})
    if config["contact"]["name"] not in names:
        raise ValueError("The contact must be one of the listed authors")
    email = text(config["contact"]["email"], "contact email")
    try:
        address = Address(addr_spec=email)
    except (TypeError, ValueError):
        raise ValueError("Invalid contact email") from None
    if not address.username or not address.domain or "." not in address.domain:
        raise ValueError("Invalid contact email")

    require_keys(config["github"], {"repo_id", "pages_url"})
    repository = config["github"]["repo_id"]
    if not isinstance(repository, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repository):
        raise ValueError("Invalid GitHub repository ID")
    if ".." in repository or repository.endswith(".git"):
        raise ValueError("Invalid GitHub repository ID")
    pages = public_url(config["github"]["pages_url"], "github.pages_url")
    if not pages.endswith("/"):
        raise ValueError("The Project Page URL must end with a slash")

    require_keys(config["dataset"], {"status", "repo_id", "revision"})
    if (config["dataset"]["status"] != "in_preparation"
            or config["dataset"]["repo_id"] is not None or config["dataset"]["revision"] is not None):
        raise ValueError("This code release requires dataset.status=in_preparation and no dataset repository or revision")
    require_keys(config["paper"], {"arxiv_id", "url", "pdf_url"}, {"primary_class", "doi"})
    paper = public_url(config["paper"]["url"], "paper.url")
    pdf = public_url(config["paper"]["pdf_url"], "paper.pdf_url")
    arxiv_id = config["paper"]["arxiv_id"]
    if arxiv_id is None:
        if set(config["paper"]) != {"arxiv_id", "url", "pdf_url"}:
            raise ValueError("arXiv citation fields require a confirmed arXiv identifier")
        if paper != pdf or not pdf.startswith(pages) or not urlsplit(pdf).path.lower().endswith(".pdf"):
            raise ValueError("The paper must link to the confirmed PDF beneath the Project Page URL")
    else:
        if not isinstance(arxiv_id, str) or not re.fullmatch(r"[0-9]{2}(?:0[1-9]|1[0-2])\.[0-9]{5}", arxiv_id):
            raise ValueError("Invalid arXiv identifier")
        if paper != f"https://arxiv.org/abs/{arxiv_id}" or pdf != f"https://arxiv.org/pdf/{arxiv_id}":
            raise ValueError("Paper URLs must match the confirmed arXiv identifier")
        primary_class = text(config["paper"].get("primary_class"), "paper.primary_class")
        if not re.fullmatch(r"[a-z][a-z0-9-]*(?:\.[A-Z][A-Z0-9-]*)?", primary_class):
            raise ValueError("Invalid arXiv primary class")
        if "doi" in config["paper"] and config["paper"]["doi"] != f"10.48550/arXiv.{arxiv_id}":
            raise ValueError("The arXiv DOI must match the confirmed identifier")
    return config


def render(config):
    institutions = {item["id"]: item["name"] for item in config["affiliations"]}
    people = [
        {"given-names": author["given_names"], "family-names": author["family_names"],
         "affiliation": "; ".join(institutions[number] for number in author["affiliations"])}
        for author in config["authors"]
    ]
    contact_index = next(i for i, author in enumerate(config["authors"])
                         if author["name"] == config["contact"]["name"])
    contact = {**people[contact_index], "email": config["contact"]["email"]}
    cff = {
        "cff-version": "1.2.0", "type": "software",
        "message": "If you use this code, please cite the accompanying paper.",
        "title": config["project"], "version": config["version"],
        "date-released": config["date"], "license": "GPL-2.0-or-later",
        "url": config["github"]["pages_url"],
        "repository-code": "https://github.com/" + config["github"]["repo_id"],
        "authors": people, "contact": [contact],
        "preferred-citation": {"type": "report", "title": config["title"],
                               "authors": people, "year": int(config["date"][:4]),
                               "url": config["paper"]["url"]},
    }
    arxiv_id = config["paper"]["arxiv_id"]
    if arxiv_id is not None:
        cff["preferred-citation"]["identifiers"] = [
            {"type": "other", "value": arxiv_id, "description": "arXiv"}
        ]
        if "doi" in config["paper"]:
            cff["preferred-citation"]["doi"] = config["paper"]["doi"]
    escaped = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "%": r"\%", "&": r"\&",
               "_": r"\_", "#": r"\#", "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

    def bib(value):
        return "".join(escaped.get(character, character) for character in value)

    authors = " and ".join(bib(author["family_names"]) + ", " + bib(author["given_names"])
                           for author in config["authors"])
    key = re.sub(r"[^A-Za-z0-9]", "", config["project"]).lower() + config["date"][:4]
    arxiv_fields = ""
    if arxiv_id is not None:
        arxiv_fields = (f"  eprint = {{{arxiv_id}}},\n"
                        "  archivePrefix = {arXiv},\n"
                        f"  primaryClass = {{{config['paper']['primary_class']}}},\n")
        if "doi" in config["paper"]:
            arxiv_fields += f"  doi = {{{config['paper']['doi']}}},\n"
    entry = (f"@misc{{{key},\n"
             f"  title = {{{{{bib(config['title'])}}}}},\n"
             f"  author = {{{authors}}},\n"
             f"  year = {{{config['date'][:4]}}},\n"
             f"{arxiv_fields}"
             f"  url = {{{config['paper']['url']}}}\n"
             "}\n")
    return {"CITATION.cff": json.dumps(cff, indent=2, ensure_ascii=False) + "\n",
            "CITATION.bib": entry}


def check_dataset_links(root, generated):
    candidates = {root / name for name in ("PUBLICATION.json", "README.md", *generated)}
    for folder in ("docs", "website/src", "website/public", "website/dist"):
        directory = root / folder
        if directory.is_dir():
            candidates.update(path for path in directory.rglob("*")
                              if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES)
    candidates.add(root / "website/index.html")
    for path in sorted(candidates):
        relative = path.relative_to(root).as_posix()
        if relative in generated:
            contents = generated[relative]
        elif path.is_file():
            if path.is_symlink():
                raise ValueError(f"Symlinked public text is not supported: {relative}")
            contents = path.read_text(encoding="utf-8")
        else:
            continue
        decoded = unquote(html.unescape(contents.replace(r"\/", "/")))
        if DATASET_LINK.search(decoded):
            raise ValueError(f"Dataset is in preparation but a Hugging Face dataset link remains in {relative}")


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check metadata, citations, and deferred-data links without writing")
    args = parser.parse_args(argv)
    root = Path(root)
    try:
        source = root / "PUBLICATION.json"
        if source.is_symlink():
            raise ValueError("PUBLICATION.json must not be a symlink")
        config = validate(json.loads(source.read_text(encoding="utf-8")))
        generated = render(config)
        check_dataset_links(root, generated if not args.check else {})
        changed = [name for name, value in generated.items()
                   if not (root / name).is_file() or (root / name).read_text(encoding="utf-8") != value]
        for name in generated:
            if (root / name).is_symlink():
                raise ValueError(f"{name} must not be a symlink")
        if args.check:
            if changed:
                raise ValueError("Generated citations are missing or outdated: " + ", ".join(changed))
            print("Publication metadata and citations are consistent. Dataset: In preparation.")
        else:
            for name in changed:
                (root / name).write_text(generated[name], encoding="utf-8")
            print("Citation files generated. Dataset: In preparation.")
        return 0
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, f"Publication check failed: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
