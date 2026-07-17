"""PlanetScope EVI validation of alfalfa cutting dates.

Independent validation of BEAST-detected cut dates (from HLS EVI) against
higher-cadence PlanetScope (~3 m) EVI, extracted cloud-native (windowed COG
reads clipped to each parcel) so no full scenes are downloaded.

Run from the repo root (evi_analysis/):
    python -m es_analysis.planet_validation.select_parcels
    PL_API_KEY=... python -m es_analysis.planet_validation.run_planet_validation --smoke
    PL_API_KEY=... python -m es_analysis.planet_validation.run_planet_validation
"""
