from edf_pipeline import run_pipeline

# Will skip conversion if .asc already exists
results = run_pipeline("[FILE_NAME].edf")

print("Fixations:")
print(results["fixations"].head())
