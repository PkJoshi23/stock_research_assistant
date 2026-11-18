"""Ingestion layer: turn external data into graph nodes, relationships and
current-snapshot properties.

Hard rule for every module in this package: write *entities*, *relationships*
and *current snapshot values* only. Never write a time series of daily
prices/open/high/low/volume - that data belongs in a time-series database and
would dilute the connected-context theme of this project.
"""
