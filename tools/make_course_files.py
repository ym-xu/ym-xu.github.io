"""Write the data files of the Python course from the simulated cohort (static/python/cohort.py).

    python tools/make_course_files.py

cohort.csv     the cohort as it might come out of a spreadsheet: one row per patient, plus a column
               wall_range (= wall_max - wall_min) that a spreadsheet formula has filled with 0 for the
               patients without wall measurements. Lesson 2 uses it to teach that missing is not zero.
followup.csv   the follow-up table (make_followup).
cohort.xlsx    both tables as sheets "patients" and "followup", for lesson 7 (needs openpyxl).
cervix.csv           the audit data of the themed practice (static/python/cervix.py) as a messy
                     spreadsheet export: one patient entered twice, the applicator written four ways,
                     and one dose with a lost decimal point. Part 2 of the practice cleans it.
cervix_followup.csv  its follow-up table (make_cervix_followup).
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

# ---- themed practice: cervical cancer brachytherapy audit ----
from cervix import make_cervix, make_cervix_followup  # noqa: E402

raw = make_cervix()
ic = raw.index[raw['technique'] == 'IC']
needles = raw.index[raw['technique'] == 'IC/IS']
raw.loc[ic[[3, 17, 40, 62, 88, 101]], 'technique'] = 'ic'            # the same applicator, written differently
raw.loc[needles[[5, 19, 33, 47, 60]], 'technique'] = 'IC+IS'
slip = raw.index[(raw['d90_3'] >= 7) & (raw['d90_3'] < 8)][10]        # a lost decimal point: 7.x entered as 7x
raw.loc[slip, 'd90_3'] = raw.loc[slip, 'd90_3'] * 10
raw = pd.concat([raw, raw.iloc[[56]]], ignore_index=True)             # one patient entered twice
raw.to_csv(HERE / 'cervix.csv', index=False)
make_cervix_followup().to_csv(HERE / 'cervix_followup.csv', index=False)
print(f'{len(raw)} rows -> cervix.csv (duplicate of {raw.iloc[-1]["id"]}, decimal slip in {raw.loc[slip, "id"]}), cervix_followup.csv')
