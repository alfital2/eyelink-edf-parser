"""
EDF to ASC Converter and ASC Parser Runner
-------------------------------------------
This module wraps the SR Research 'edf2asc' tool to convert EDF to ASC,
then parses the ASC file using the EyeLinkParser class.

Author: TAL ALFI
Date: 2025-03-22
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict
import pandas as pd

from eyelink_parser import EyeLinkParser

class EDFConverter:
    """
    A converter class to wrap the SR Research 'edf2asc' binary tool.
    Converts EyeLink .edf files to .asc format.
    """

    def __init__(self, edf2asc_path: str = "/Applications/EyeLink DataViewer 4.4/EDFConverter.app/Contents/MacOS/EDFConverter"):
        """
        Initialize the converter.

        :param edf2asc_path: Path to the edf2asc executable
        """
        if not Path(edf2asc_path).exists():
            raise FileNotFoundError(f"edf2asc tool not found at '{edf2asc_path}'")

        self.edf2asc_path = edf2asc_path

    def convert(self, edf_path: str, output_dir: Optional[str] = None, overwrite: bool = True) -> str:
        """
        Convert an EDF file to ASC using edf2asc.

        :param edf_path: Path to the .edf file
        :param output_dir: Optional output directory for the .asc file
        :param overwrite: If True, overwrite existing .asc file
        :return: Path to the generated .asc file
        """
        edf_path = Path(edf_path)
        if not edf_path.exists():
            raise FileNotFoundError(f"EDF file '{edf_path}' does not exist")

        output_dir = Path(output_dir) if output_dir else edf_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        asc_path = output_dir / (edf_path.stem + ".asc")
        if asc_path.exists() and not overwrite:
            return str(asc_path)

        cmd = [
            self.edf2asc_path,
            "-e",  # Include events
            "-s",  # Include samples
            "-n"   # No screen output
        ]

        if overwrite:
            cmd.append("-y")  # Auto-confirm overwrite

        cmd.append(str(edf_path))

        try:
            subprocess.run(cmd, cwd=output_dir, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"edf2asc failed: {e}")

        if not asc_path.exists():
            raise FileNotFoundError(f"Conversion failed. Expected ASC file not found: {asc_path}")

        return str(asc_path)

def run_pipeline(edf_path: str, edf2asc_path: str = "/Applications/EyeLink DataViewer 4.4/EDFConverter.app/Contents/MacOS/EDFConverter", force_convert: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Convert an EDF file to ASC (unless ASC already exists), and parse into structured DataFrames.

    :param edf_path: Path to the EDF file
    :param edf2asc_path: Optional custom path to the edf2asc binary
    :param force_convert: If True, forces EDF to ASC conversion even if ASC exists
    :return: A dictionary of DataFrames: fixations, saccades, blinks, messages
    """
    edf_path = Path(edf_path)
    asc_path = edf_path.with_suffix(".asc")

    if asc_path.exists() and not force_convert:
        print(f"ASC file already exists. Skipping conversion: {asc_path.name}")
    else:
        converter = EDFConverter(edf2asc_path=edf2asc_path)
        asc_path = converter.convert(str(edf_path), overwrite=force_convert)

    parser = EyeLinkParser(str(asc_path))
    parser.parse()

    return {
        "fixations": parser.get_fixations(),
        "saccades": parser.get_saccades(),
        "blinks": parser.get_blinks(),
        "messages": parser.get_messages()
    }

# Example usage:
# results = run_pipeline("path/to/session.edf")
# print(results["fixations"].head())
