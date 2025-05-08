import sys
import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional

# Add parent directory to path to import from the root parser
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import functions from the root parser
from parser import (
    process_asc_file, process_multiple_files, 
    load_csv_file, load_multiple_csv_files,
    EyeLinkASCParser
)