import traceback
import xarray as xr
import numpy as np
import os
import os.path
import re
from datetime import datetime
import argparse
from pathlib import Path

import wrf_tools
import wrf_AR_analysis
import wrf_interpolate_pressure

def check(filename, varnames):
    
    ds = xr.open_dataset(filename)
    ds_varnames = list(ds.keys)
    
    result = {
        varname : varname in ds_varnames
        for varname in varnames
    }

    return result





pressure_levs = [1000, 925, 850, 700, 500, 300, 200, 100, 50, 10]

def doWork(details):
    
    
    result = dict(details=details, status="UNKNOWN")

    input_file = Path(details["input_file"])
    output_file = Path(details["output_file"])
    
    try:


        output_dir = output_file.parents[0]
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Input file: ", input_file)
        ds_original = xr.open_dataset(input_file, engine="netcdf4")
        ds_diag = generateReduction(ds_original)

        print("Writing output: ", output_file)
        ds_diag.to_netcdf(output_file)

        result['status'] = "OK"

    except Exception as e:

        result['status'] = 'ERROR'
        traceback.print_exc()
        print(e)



    return result



def generateReduction(
    ds,
):
    merge_data = []
 
    for varname in ds.keys():
    
        da = ds[varname]
        da_dims = da.dims
    
        if ("bottom_top" not in da_dims) and ("bottom_top_stag" not in da_dims):
            merge_data.append(da) 

    merge_data.append(wrf_interpolate_pressure.generatePressureDiag(ds, p = pressure_levs))
    merge_data.append(wrf_AR_analysis.generateARDiag(ds))
    
    new_ds = xr.merge(merge_data)

    return new_ds


