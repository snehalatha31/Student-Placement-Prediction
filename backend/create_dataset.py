from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "placement_data.csv"

np.random.seed(42)

N = 1200

cgpa = np.round(
    np.random.uniform(5.0, 10.0, N), 2
)

tenth = np.round(
    np.random.uniform(50, 100, N), 2
)

twelfth = np.round(
    np.random.uniform(50, 100, N), 2
)

backlogs = np.random.randint(0, 5, N)

attendance = np.round(
    np.random.uniform(55, 100, N), 2
)

internship = np.random.randint(0, 2, N)

projects = np.random.randint(0, 5, N)

communication = np.random.randint(1, 11, N)

technical = np.random.randint(1, 11, N)


score = (
    cgpa * 10
    + tenth * 0.20
    + twelfth * 0.20
    + attendance * 0.15
    + internship * 8
    + projects * 5
    + communication * 2
    + technical * 3
    - backlogs * 12
)


eligible = (
    score >= 125
).astype(int)


data = pd.DataFrame({

    "CGPA": cgpa,

    "Tenth_Percentage": tenth,

    "Twelfth_Percentage": twelfth,

    "Backlogs": backlogs,

    "Attendance": attendance,

    "Internship": internship,

    "Projects": projects,

    "Communication_Skill": communication,

    "Technical_Skill": technical,

    "Eligible": eligible
})


DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True
)

data.to_csv(
    DATASET_PATH,
    index=False
)


print("Dataset created successfully.")

print(f"Saved to: {DATASET_PATH}")

print(f"Rows: {len(data)}")

print(data.head())