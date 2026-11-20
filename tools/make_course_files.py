"""Write the data files of the Python course from the simulated cohort (static/python/cohort.py).

    python tools/make_course_files.py

cohort.csv     the cohort as it might come out of a spreadsheet: one row per patient, plus a column
               wall_range (= wall_max - wall_min) that a spreadsheet formula has filled with 0 for the
               patients without wall measurements. Lesson 2 uses it to teach that missing is not zero.
followup.csv   the follow-up table (make_followup).
cohort.xlsx    both tables as sheets "patients" and "followup", for lesson 7 (needs openpyxl).
All values are simulated; no file here contains patient data.
"""
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1] / 'static' / 'python'
sys.path.insert(0, str(HERE))
from cohort import make_cohort, make_followup  # noqa: E402

df = make_cohort()
wall_range = (df['wall_max'] - df['wall_min']).round(1).fillna(0)      # what the spreadsheet formula gives for empty cells
df.insert(df.columns.get_loc('wall_max') + 1, 'wall_range', wall_range)
fu = make_followup()

df.to_csv(HERE / 'cohort.csv', index=False)
fu.to_csv(HERE / 'followup.csv', index=False)
with pd.ExcelWriter(HERE / 'cohort.xlsx') as xl:
    df.to_excel(xl, sheet_name='patients', index=False)
    fu.to_excel(xl, sheet_name='followup', index=False)
print(f'{len(df)} patients, {len(fu)} with follow-up -> cohort.csv, followup.csv, cohort.xlsx in {HERE}')
