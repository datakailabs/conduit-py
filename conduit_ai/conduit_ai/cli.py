"""Conduit CLI — install, inspect, and query knowledge packs."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

from conduit_ai.client import ConduitClient


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="conduit",
        description="Conduit knowledge graph engine CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # ─── install ─────────────────────────────────────────────────
    install_p = sub.add_parser("install", help="Install a knowledge pack")
    install_p.add_argument("pack", help="Pack file (.ckp) or registry ID")
    install_p.add_argument("--topics", help="Comma-separated topic filter", default=None)
    install_p.add_argument("--endpoint", default=os.getenv("CONDUIT_ENDPOINT", "http://localhost:4000"))
    install_p.add_argument("--api-key", default=os.getenv("CONDUIT_ADMIN_KEY", ""))
    install_p.add_argument("--org", default="org_datakai")
    install_p.add_argument("--dry-run", action="store_true", help="Show what would be installed without ingesting")

    # ─── inspect ─────────────────────────────────────────────────
    inspect_p = sub.add_parser("inspect", help="Inspect a knowledge pack")
    inspect_p.add_argument("pack", help="Pack file (.ckp)")
    inspect_p.add_argument("--topics", action="store_true", help="Show full topic list")

    # ─── ask ─────────────────────────────────────────────────────
    ask_p = sub.add_parser("ask", help="Ask the knowledge graph a question")
    ask_p.add_argument("query", help="Question to ask")
    ask_p.add_argument("--limit", type=int, default=8)
    ask_p.add_argument("--endpoint", default=os.getenv("CONDUIT_ENDPOINT", "http://localhost:4000"))
    ask_p.add_argument("--api-key", default=os.getenv("CONDUIT_API_KEY", ""))

    # ─── list ────────────────────────────────────────────────────
    sub.add_parser("list", help="List installed packs (reads from knowledge graph domains)")

    args = parser.parse_args()

    if args.command == "install":
        cmd_install(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect a .ckp pack file."""
    pack_path = Path(args.pack)
    if not pack_path.exists():
        print(f"Error: {pack_path} not found", file=sys.stderr)
        sys.exit(1)

    with tarfile.open(pack_path, "r:gz") as tar:
        # Read manifest
        manifest_f = tar.extractfile("pack.toml")
        if not manifest_f:
            print("Error: pack.toml not found in archive", file=sys.stderr)
            sys.exit(1)

        manifest = manifest_f.read().decode("utf-8")

        # Parse TOML (minimal parser — just extract key fields)
        pack_id = _toml_get(manifest, "id")
        name = _toml_get(manifest, "name")
        version = _toml_get(manifest, "version")
        description = _toml_get(manifest, "description")
        zettels = _toml_get(manifest, "zettels")
        relationships = _toml_get(manifest, "relationships")
        domains = _toml_get(manifest, "domains")

        print(f"\n  {name} v{version}")
        print(f"  {description}")
        print(f"  {zettels} zettels, {relationships} relationships")
        if domains:
            print(f"  Domains: {domains}")
        print()

        # Show topics
        in_topics = False
        topics = []
        for line in manifest.split("\n"):
            if line.strip() == "[pack.topics]":
                in_topics = True
                continue
            if in_topics:
                if line.startswith("["):
                    break
                if "=" in line:
                    parts = line.split("=", 1)
                    topics.append((parts[0].strip(), parts[1].strip()))

        if topics:
            shown = topics if args.topics else topics[:15]
            print(f"  Topics ({len(topics)}):")
            max_name = max(len(t[0]) for t in shown)
            for name, count in shown:
                dots = "." * (max_name - len(name) + 3)
                print(f"    {name} {dots} {count} zettels")
            if not args.topics and len(topics) > 15:
                print(f"    ... and {len(topics) - 15} more (use --topics to see all)")
        print()

        print(f"  Install full:   conduit install {pack_path}")
        if topics:
            top_3 = ",".join(t[0] for t in topics[:3])
            print(f"  Install scoped: conduit install {pack_path} --topics {top_3}")
        print()


def cmd_install(args: argparse.Namespace) -> None:
    """Install a .ckp pack into a Conduit instance."""
    pack_path = Path(args.pack)
    if not pack_path.exists():
        print(f"Error: {pack_path} not found", file=sys.stderr)
        sys.exit(1)

    topic_filter = set(args.topics.split(",")) if args.topics else None

    with tarfile.open(pack_path, "r:gz") as tar:
        # Read zettels
        zettels_f = tar.extractfile("zettels.jsonl")
        if not zettels_f:
            print("Error: zettels.jsonl not found", file=sys.stderr)
            sys.exit(1)

        zettels = []
        for line in zettels_f:
            z = json.loads(line)
            if topic_filter:
                zettel_topics = set(z.get("topics", []))
                if not zettel_topics & topic_filter:
                    continue
            zettels.append(z)

        print(f"Pack: {pack_path.name}")
        print(f"Zettels to install: {len(zettels)}")
        if topic_filter:
            print(f"Topic filter: {', '.join(sorted(topic_filter))}")

        if args.dry_run:
            print("\n[DRY RUN] Would install the above. Use without --dry-run to proceed.")
            return

        if not args.api_key:
            print("Error: --api-key or CONDUIT_ADMIN_KEY required for install", file=sys.stderr)
            sys.exit(1)

        # Ingest via extract API (batch by combining zettels into text chunks)
        client = ConduitClient(api_key=args.api_key, endpoint=args.endpoint)
        installed = 0
        failed = 0

        print(f"\nInstalling into {args.endpoint}...")

        # Process in batches of 5 zettels (to stay within extract API limits)
        batch_size = 5
        for i in range(0, len(zettels), batch_size):
            batch = zettels[i : i + batch_size]
            combined_text = "\n\n---\n\n".join(
                f"# {z['title']}\n\n{z['content']}" for z in batch
            )

            try:
                import httpx

                resp = httpx.post(
                    f"{args.endpoint}/api/v1/extract",
                    headers={"Authorization": f"Bearer {args.api_key}"},
                    json={
                        "text": combined_text,
                        "contextSource": batch[0].get("context_source", "vendor-doc"),
                        "domainHints": batch[0].get("domains", []),
                        "maxUnits": len(batch) * 3,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                result = resp.json()
                installed += result.get("novel", 0)
                progress = min(i + batch_size, len(zettels))
                print(f"  [{progress}/{len(zettels)}] +{result.get('novel', 0)} new, {result.get('duplicates', 0)} dupes")
            except Exception as e:
                failed += len(batch)
                print(f"  [{i}/{len(zettels)}] Error: {e}", file=sys.stderr)

        print(f"\nInstalled: {installed} knowledge units")
        if failed:
            print(f"Failed: {failed} zettels")


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question via the CLI."""
    client = ConduitClient(api_key=args.api_key, endpoint=args.endpoint)
    resp = client.ask(args.query, limit=args.limit)

    print(f"\n{resp.answer}\n")
    if resp.sources:
        print(f"Sources ({len(resp.sources)}):")
        for s in resp.sources:
            print(f"  - {s.title} ({s.score:.3f}) {s.domains}")
    print()


def cmd_list(args: argparse.Namespace) -> None:
    """List domains in the knowledge graph (proxy for installed packs)."""
    endpoint = os.getenv("CONDUIT_ENDPOINT", "http://localhost:4000")
    api_key = os.getenv("CONDUIT_API_KEY", "")

    import httpx

    resp = httpx.post(
        f"{endpoint}/graphql",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": "{ topology { domains { domain count } } }"},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    domains = data.get("data", {}).get("topology", {}).get("domains", [])
    if not domains:
        print("No knowledge installed.")
        return

    print("\nInstalled knowledge domains:\n")
    for d in sorted(domains, key=lambda x: x["count"], reverse=True):
        print(f"  {d['domain']:20s} {d['count']:>6,} zettels")
    print()


def _toml_get(text: str, key: str) -> str:
    """Minimal TOML value extractor — works for simple key = value lines."""
    for line in text.split("\n"):
        if line.strip().startswith(f"{key} =") or line.strip().startswith(f"{key}="):
            val = line.split("=", 1)[1].strip()
            return val.strip('"').strip("'")
    return ""


if __name__ == "__main__":
    main()
