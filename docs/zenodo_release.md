# Zenodo data-release plan

The code lives on GitHub (`ucm-vista/alfalfa-evi-water-savings`). The analysis data
is too large for git (~4 GB of CSVs, several files over GitHub's 100 MB hard limit),
so it is published separately on **Zenodo**, which is built for multi-GB research data
and issues a citable **DOI**. This file is the checklist to make that release.

## What to deposit

The repo is already wired for a **curated intermediate bundle (~207 MB)** — this is the
recommended scope. It is the minimum needed to reproduce every downstream figure/table
*without* the raw rebuild or the slow R/BEAST stage. Contents (see `data/README.md`):

| Component | Size | Role |
|-----------|------|------|
| `beast_outputs_new/` | 15 MB | BEAST cutting-detection results |
| `county_year_exports_new/` | 167 MB | per-county/water-year parcel EVI exports (BEAST inputs) |
| `water_saving_scenarios_stats/` | 4.9 MB | late-cut savings tables |
| `Cutting_weather_stats/` | 4.2 MB | merged cut + weather stats |
| `statistics_exports/` | 16 MB | precomputed statistics tables |

**Optional second archive — full analysis CSVs (~4 GB).** If the colleague wants the
complete CSV set (all per-parcel EVI/NDWI timeseries, `evi_daily_processed.csv`, etc.),
add it as a *separate* file in the same Zenodo record — e.g. `analysis_csv_full_v1.tar.gz`.
Keep it separate so casual users can grab the 207 MB bundle without the 4 GB download.
Before archiving, prune the redundancy we found: drop every `.ipynb_checkpoints/` copy and
de-duplicate the parallel `evi_analysis/` vs `evi_analysis/reports/` trees — that removes
~1 GB of exact duplicates.

## Steps

1. **Stage the bundle** into `release_assets/` (paths inside the tarball are relative to
   the repo root, matching what `fetch_intermediate.sh` expects on extract):
   ```bash
   cd "$WORK_ROOT"        # /home/jovyan/work
   tar -czf alfalfa-evi-water-savings/release_assets/intermediate_data_v1.tar.gz \
       beast_outputs_new county_year_exports_new \
       water_saving_scenarios_stats Cutting_weather_stats statistics_exports
   # NOTE: adjust source paths to wherever these dirs actually live under $WORK_ROOT.
   ```
2. **Checksum** it and record the value in `release_assets/MANIFEST.md`:
   ```bash
   sha256sum release_assets/intermediate_data_v1.tar.gz
   ```
3. **Create the Zenodo record** (https://zenodo.org, log in with ORCID/GitHub):
   upload the tarball(s), set title = the paper's, authors = Silberman + colleague(s),
   license = MIT (or CC-BY-4.0 for the data), link the GitHub repo URL in "Related identifiers".
   Save a **draft** first to reserve a DOI without publishing.
4. **Publish** to mint the DOI. Zenodo gives a versioned DOI plus a concept DOI that always
   points to the latest version.
5. **Wire the DOI back into the repo** (three edits, one commit):
   - `scripts/fetch_intermediate.sh` → replace the `XXXXXXX` in the `URL=` line with the real record.
   - `CITATION.cff` → uncomment `doi:` and fill in `10.5281/zenodo.XXXXXXX`.
   - `README.md` → the "Fast path" Zenodo line, add the record URL.
   Then: `git commit -am "Add Zenodo DOI for intermediate-data release" && git push`.

## Notes

- Zenodo free deposits are up to **50 GB per record**, so both archives fit comfortably.
- Tie the release to a git tag/GitHub Release (e.g. `v1.0.0`) so code + data versions line up.
  Zenodo's GitHub integration can auto-archive a tagged release if you enable the repo there.
- The DWR i15 crop-mapping shapefile stays in the git repo (7.4 MB, public); it is not part of
  the Zenodo bundle.
