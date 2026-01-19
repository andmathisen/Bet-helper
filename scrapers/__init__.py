"""Scrapers module for web scraping football data.

Keep this module import-light.

Some scraper implementations pull in Selenium + webdriver_manager (and transitively networking/SSL),
so we avoid importing those at package import time. Import concrete scraper classes from their
modules directly (e.g. `from scrapers.footystats_scraper import FootyStatsScraper`).
"""

from scrapers.base_scraper import BaseScraper
from scrapers.league_mapping import LEAGUE_CODE_TO_PATH, code_to_path, path_to_code

__all__ = ["BaseScraper", "code_to_path", "path_to_code", "LEAGUE_CODE_TO_PATH"]