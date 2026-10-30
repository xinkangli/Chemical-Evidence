"""Synthetic input for software demonstration only, not validation data."""
from pathlib import Path
import numpy as np
p=Path('outputs');p.mkdir(exist_ok=True)
x=np.arange(600.,3301.);y=.01+sum(np.exp(-.5*((x-c)/4)**2) for c in [2915,2845,1462,717])
np.savetxt(p/'synthetic_pe.csv',np.column_stack([x,y]),delimiter=',',header='wavenumber,intensity',comments='')
