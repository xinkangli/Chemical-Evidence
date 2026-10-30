"""Inspect a CSV spectrum without generating or certifying an identity."""
import argparse,json
from pathlib import Path
import numpy as np
from .bands import check_spectrum,summarize_candidate,SIGNATURES
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('csv',help='CSV with wavenumber,intensity header')
 p.add_argument('--candidate',choices=sorted(SIGNATURES),required=True)
 p.add_argument('--mode',choices=['ATR','Transmission','unknown'],required=True)
 p.add_argument('--signal-type',choices=['absorbance','reported_absorbance','transmittance_fraction','transmittance_percent','unknown'],required=True)
 p.add_argument('--id',default='user-spectrum');p.add_argument('--output',required=True)
 a=p.parse_args();data=np.genfromtxt(a.csv,delimiter=',',names=True)
 packet=check_spectrum(data['wavenumber'],data['intensity'],a.mode,a.signal_type,a.id)
 summary=summarize_candidate([packet],a.candidate)
 summary['chemical_support']=summary['unique_peak_observations']/summary['expected_checks']
 Path(a.output).write_text(json.dumps({'measurement':packet,'candidate':summary},indent=2,allow_nan=False))
if __name__=='__main__':main()
