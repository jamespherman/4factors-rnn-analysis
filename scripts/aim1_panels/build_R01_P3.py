#!/usr/bin/env python3
"""
Build R01 panel P3: % units with pre-saccadic suppression by target angle from the RF centre, classic (E) and
interneurons (I). Values are taken unchanged from results/aim1_panels/B4_map_competition/ (B4_suppression.csv and
the McNemar rows of stats.csv), which build_B4.py computed from results/intercollicular/g1_geometry/rows.npy
(scripts/ic_g1_geometry.py): presac window -50..0 ms re saccade onset vs pre-target -200..0 ms, paired t-test per
target location, p < 0.01 and negative; Wilson 95% CI; exact McNemar on discordant units.
Run: python scripts/aim1_panels/build_R01_P3.py   (seconds)
"""
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / 'results' / 'aim1_panels' / 'B4_map_competition'
OUT = REPO / 'results' / 'aim1_panels' / 'R01_core' / 'P3_suppression_by_angle'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
    bars = pd.read_csv(SRC / 'B4_suppression.csv')
    bars.to_csv(OUT / 'P3_suppression.csv', index=False, float_format='%.5g')
    st = pd.read_csv(SRC / 'stats.csv')
    st = st[st.statistic.str.startswith('presac suppression')].copy()
    st['window'] = 'presac -50..0 ms re saccade onset vs pre-target -200..0 ms; p<0.01 per location'
    st['session_filter'] = '6 diagonal-target sessions (Newton 08_15 excluded), all units (g1_geometry)'
    st['source'] = 'results/aim1_panels/B4_map_competition/stats.csv (build_B4.py; ic_g1_geometry.py) via build_R01_P3.py'
    st.to_csv(OUT / 'stats.csv', index=False, float_format='%.5g')
    print(bars.round(1).to_string())
    print(st[['statistic', 'unit_class', 'value', 'p', 'n']].to_string())


if __name__ == '__main__':
    main()
