"""Runner: OpenET direct API downloader.

Usage:
    python -m es_analysis.runners.run_openet_download download [options]
    python -m es_analysis.runners.run_openet_download assemble [options]
    python -m es_analysis.runners.run_openet_download validate [options]
    python -m es_analysis.runners.run_openet_download status [options]

Downloads parcel-level ET timeseries from the OpenET REST API,
assembles raw JSON into year-partitioned CSVs, and validates completeness.

Parallel download (2 keys, 4 streams splitting by county):
    # Terminal 1
    python -m es_analysis.runners.run_openet_download download \
        --counties San_Joaquin Stanislaus Merced --api-key KEY1 -v
    # Terminal 2
    python -m es_analysis.runners.run_openet_download download \
        --counties Madera Fresno Kings --api-key KEY1 -v
    # Terminal 3
    python -m es_analysis.runners.run_openet_download download \
        --counties Tulare Kern --api-key KEY2 -v
    # Terminal 4
    python -m es_analysis.runners.run_openet_download download \
        --counties Riverside Imperial --api-key KEY2 -v
"""

import argparse
import logging
import sys


def _run_download(args: argparse.Namespace) -> None:
    """Execute the download subcommand."""
    from es_analysis.data_providers.openet_downloader import run_download
    from pathlib import Path

    kwargs = dict(
        max_workers=args.max_workers,
        delay=args.delay,
        dry_run=args.dry_run,
        interval=args.interval,
        verbose=args.verbose,
    )
    if args.counties:
        kwargs["counties"] = args.counties
    if args.out_dir:
        kwargs["download_root"] = Path(args.out_dir)
    if args.max_requests is not None:
        kwargs["max_requests"] = args.max_requests
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.api_keys:
        kwargs["api_keys"] = args.api_keys

    run_download(**kwargs)


def _run_assemble(args: argparse.Namespace) -> None:
    """Execute the assemble subcommand."""
    from es_analysis.data_providers.openet_downloader import assemble_csvs
    from pathlib import Path

    kwargs = dict(interval=args.interval)
    if args.download_dir:
        kwargs["download_root"] = Path(args.download_dir)
    if args.output_dir:
        kwargs["output_root"] = Path(args.output_dir)

    assemble_csvs(**kwargs)


def _run_validate(args: argparse.Namespace) -> None:
    """Execute the validate subcommand."""
    from es_analysis.data_providers.openet_downloader import validate_download
    from pathlib import Path

    kwargs = dict(interval=args.interval)
    if args.download_dir:
        kwargs["download_root"] = Path(args.download_dir)
    if args.counties:
        kwargs["counties"] = args.counties

    df = validate_download(**kwargs)
    print("\nDownload Completeness Report:")
    print(df.to_string(index=False))
    total = df["total_parcels"].sum()
    done = df["completed"].sum()
    pct = round(100 * done / total, 1) if total > 0 else 0
    print(f"\nOverall: {done}/{total} parcels ({pct}%)")


def _run_status(args: argparse.Namespace) -> None:
    """Execute the status subcommand."""
    from es_analysis.data_providers.openet_downloader import get_status
    from pathlib import Path

    kwargs = {}
    if args.download_dir:
        kwargs["download_root"] = Path(args.download_dir)

    status = get_status(**kwargs)
    print("\nOpenET Download Status:")
    for interval, info in status.items():
        if isinstance(info, dict):
            print(f"  {interval}:")
            for k, v in info.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {interval}: {info}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenET direct API downloader for parcel-level ET.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- download ---
    dl = subparsers.add_parser("download", help="Download ET data from OpenET API.")
    dl.add_argument("--counties", nargs="+", default=None,
                    help="County names to download (default: all).")
    dl.add_argument("--max-requests", type=int, default=None,
                    help="Stop after N API calls (for quota management).")
    dl.add_argument("--max-workers", type=int, default=1,
                    help="Number of parallel workers (default: 1).")
    dl.add_argument("--delay", type=float, default=3.0,
                    help="Seconds between requests (default: 3.0).")
    dl.add_argument("--out-dir", type=str, default=None,
                    help="Download root directory.")
    dl.add_argument("--interval", choices=["daily", "monthly"], default="daily",
                    help="Temporal interval (default: daily).")
    dl.add_argument("--api-key", type=str, default=None,
                    help="OpenET API key (overrides config).")
    dl.add_argument("--api-keys", nargs="+", default=None,
                    help="Multiple API keys — rotates on 429.")
    dl.add_argument("--dry-run", action="store_true",
                    help="Show queue without downloading.")
    dl.add_argument("-v", "--verbose", action="store_true",
                    help="Verbose logging.")

    # --- assemble ---
    asm = subparsers.add_parser("assemble", help="Assemble raw JSON into year CSVs.")
    asm.add_argument("--download-dir", type=str, default=None,
                     help="Download root directory.")
    asm.add_argument("--output-dir", type=str, default=None,
                     help="Output directory for CSVs (default: same as download-dir).")
    asm.add_argument("--interval", choices=["daily", "monthly"], default="daily",
                     help="Interval to assemble (default: daily).")

    # --- validate ---
    val = subparsers.add_parser("validate", help="Validate download completeness.")
    val.add_argument("--download-dir", type=str, default=None,
                     help="Download root directory.")
    val.add_argument("--counties", nargs="+", default=None,
                     help="County names to validate (default: all).")
    val.add_argument("--interval", choices=["daily", "monthly"], default="daily",
                     help="Interval to validate (default: daily).")

    # --- status ---
    st = subparsers.add_parser("status", help="Show download progress.")
    st.add_argument("--download-dir", type=str, default=None,
                    help="Download root directory.")

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "download":
        _run_download(args)
    elif args.command == "assemble":
        _run_assemble(args)
    elif args.command == "validate":
        _run_validate(args)
    elif args.command == "status":
        _run_status(args)


if __name__ == "__main__":
    main()
