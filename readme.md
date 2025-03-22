# EyeLink EDF to ASC Parser Pipeline

This project provides a streamlined way to convert proprietary `.edf` eye-tracking files from SR Research's EyeLink system into readable `.asc` text files, and parse them into structured `pandas` DataFrames. It supports fixations, saccades, blinks, and messages.

---

## 🔍 What This Does

1. **Converts `.edf` to `.asc`** using the official EyeLink `edf2asc` tool (macOS & Windows compatible)
2. **Parses `.asc` files** to extract:
   - Fixation events
   - Saccade events
   - Blink events
   - Message logs
3. **Returns DataFrames** for easy inspection and analysis

---

## 📁 Project Structure

```
project_root/
├── edf_pipeline.py          # Conversion + parsing pipeline
├── eyelink_parser.py        # ASC event parser
├── main.py                  # Example script to run the pipeline
├── [FILE_NAME].edf            # Sample EDF file (not included)
├── [FILE_NAME].asc            # Sample ASC output (optional)
└── README.md
```

---

## ⚙️ Prerequisites

### Python
- Python 3.7+
- `pandas`

Install with:
```bash
pip install pandas
```

### EyeLink `edf2asc` Tool
You **must install the official `edf2asc` utility** provided by SR Research.

#### 📦 On macOS:
- Install the **EyeLink DataViewer** or **EyeLink Developers Kit**
- Locate the CLI tool (usually):
  ```
  /Applications/EyeLink DataViewer 4.4/EDFConverter.app/Contents/MacOS/EDFConverter
  ```

#### 📦 On Windows:
- The `edf2asc.exe` tool is typically located at:
  ```
  C:\Program Files (x86)\SR Research\EyeLink\EDF_Access_API\edf2asc.exe
  ```

---

## 🚀 How to Use

```python
from edf_pipeline import run_pipeline

# Will skip conversion if .asc file already exists
results = run_pipeline("[FILE_NAME].edf")

print("Fixations:")
print(results["fixations"].head())
```

To force reconversion:
```python
results = run_pipeline("[FILE_NAME].edf", force_convert=True)
```

---

## 🧪 Output
Each key in the returned dictionary is a `pandas.DataFrame`:
- `fixations`
- `saccades`
- `blinks`
- `messages`

You can inspect them using `.head()` or export them with `.to_csv()`.

---

## ✅ Example Output
```plaintext
Fixations:
   event_type eye  start_time  end_time  duration  x_mean  y_mean  pupil_size
0   fixation   L    100000     100300     300      640.2   512.1     841.3
...
```

---

## 📌 License
MIT License.

---

## 👤 Author
**Tal Alfi**  
Created as part of an eye-tracking data pipeline project.

---

## 📬 Questions / Issues
Feel free to open an issue or contact me if you'd like to contribute or request improvements.

