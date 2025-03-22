"""
EyeLink ASC File Parser
------------------------
This module provides a clean and extensible parser for EyeLink `.asc` files.
It extracts structured data from eye-tracking events including fixations,
saccades, blinks, and custom messages.

Author: TAL ALFI
Date: 2025-03-22
"""

import re
import pandas as pd
from typing import List, Dict, Optional


class EyeLinkParser:
    """
    A parser for EyeLink .asc text files.
    Parses eye-tracking events: fixations, saccades, blinks, and messages.
    """

    def __init__(self, filepath: str):
        """
        Initialize the parser with the path to the .asc file.

        :param filepath: Path to the EyeLink .asc file
        """
        self.filepath = filepath
        self.fixations: List[Dict] = []
        self.saccades: List[Dict] = []
        self.blinks: List[Dict] = []
        self.messages: List[Dict] = []

    def parse(self):
        """
        Read and parse the .asc file line by line.
        Populates internal lists with parsed event dictionaries.
        """
        with open(self.filepath, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('EFIX'):
                    self.fixations.append(self._parse_efix(line))
                elif line.startswith('ESACC'):
                    self.saccades.append(self._parse_esacc(line))
                elif line.startswith('EBLINK'):
                    self.blinks.append(self._parse_eblink(line))
                elif line.startswith('MSG'):
                    parsed = self._parse_msg(line)
                    if parsed:
                        self.messages.append(parsed)

    def _parse_efix(self, line: str) -> Dict:
        parts = line.strip().split()
        return {
            'event_type': 'fixation',
            'eye': parts[1],
            'start_time': int(parts[2]),
            'end_time': int(parts[3]),
            'duration': int(parts[4]),
            'x_mean': float(parts[5]),
            'y_mean': float(parts[6]),
            'pupil_size': float(parts[7])
        }

    def _parse_esacc(self, line: str) -> Dict:
        parts = line.strip().split()
        return {
            'event_type': 'saccade',
            'eye': parts[1],
            'start_time': int(parts[2]),
            'end_time': int(parts[3]),
            'duration': int(parts[4]),
            'x_start': float(parts[5]),
            'y_start': float(parts[6]),
            'x_end': float(parts[7]),
            'y_end': float(parts[8]),
            'amplitude': float(parts[9]),
            'peak_velocity': float(parts[10])
        }

    def _parse_eblink(self, line: str) -> Dict:
        parts = line.strip().split()
        return {
            'event_type': 'blink',
            'eye': parts[1],
            'start_time': int(parts[2]),
            'end_time': int(parts[3]),
            'duration': int(parts[4])
        }

    def _parse_msg(self, line: str) -> Optional[Dict]:
        match = re.match(r"MSG\s+(\d+)\s+(.*)", line.strip())
        if match:
            return {
                'event_type': 'message',
                'timestamp': int(match.group(1)),
                'msg_text': match.group(2)
            }
        return None

    def get_fixations(self) -> pd.DataFrame:
        """Return fixations as a pandas DataFrame."""
        return pd.DataFrame(self.fixations)

    def get_saccades(self) -> pd.DataFrame:
        """Return saccades as a pandas DataFrame."""
        return pd.DataFrame(self.saccades)

    def get_blinks(self) -> pd.DataFrame:
        """Return blinks as a pandas DataFrame."""
        return pd.DataFrame(self.blinks)

    def get_messages(self) -> pd.DataFrame:
        """Return messages as a pandas DataFrame."""
        return pd.DataFrame(self.messages)

    def summary(self):
        """
        Print a quick summary of the parsed data.
        """
        print(f"Fixations: {len(self.fixations)}")
        print(f"Saccades: {len(self.saccades)}")
        print(f"Blinks: {len(self.blinks)}")
        print(f"Messages: {len(self.messages)}")


# Example usage:
parser = EyeLinkParser("185988598.asc")
parser.parse()
df_fix = parser.get_fixations()
df_sac = parser.get_saccades()
df_blink = parser.get_blinks()
df_msg = parser.get_messages()
parser.summary()
