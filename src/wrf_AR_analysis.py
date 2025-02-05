import xarray as xr
import numpy as np
import os
import os.path
import re
from datetime import datetime

import wrf_tools


def generateARDiag(
    ds,
):
    
    merge_data = []

    # Compute V_T
    _tmp = ds["V"].to_numpy()
    V_T = xr.zeros_like(ds["T"]).rename("V_T")
    V_T[:, :, :, :] = (_tmp[:, :, 1:, :] + _tmp[:, :, :-1, :]) / 2
    V_T = V_T.rename("V_T")
 

    # Compute U_T
    _tmp = ds["U"].to_numpy()
    U_T = xr.zeros_like(ds["T"]).rename("U_T")
    U_T[:, :, :, :] = (_tmp[:, :, :, 1:] + _tmp[:, :, :, :-1]) / 2
    U_T = U_T.rename("U_T")
 
    # IWV
    IWV = wrf_tools.integrateVertically(ds["QVAPOR"], ds, avg=False).rename("IWV")
 
    # IVT
    IVT_x = wrf_tools.integrateVertically(ds["QVAPOR"] * U_T, ds, avg=False)
    IVT_y = wrf_tools.integrateVertically(ds["QVAPOR"] * V_T, ds, avg=False)
    
    # code order matters here
    IVT = ((IVT_x**2 + IVT_y**2)**0.5).rename("IVT")
    IVT_x = IVT_x.rename("IVT_x")
    IVT_y = IVT_y.rename("IVT_y")

    IVT = IVT.assign_attrs(
        units = "kg m^-1 s^-1",
        description = "Magnitude of integrated vapor transport.",
    )
 
    IVT_x = IVT_x.assign_attrs(
        units = "kg m^-1 s^-1",
        description = "X-direction of integrated vapor transport.",
    )
 
    IVT_y = IVT_y.assign_attrs(
        units = "kg m^-1 s^-1",
        description = "Y-direction of integrated vapor transport.",
    )
 
    IWV = IWV.assign_attrs(
        units = "kg m^-2",
        description = "Integrated water vapor.",
    )
    

    merge_data.extend([IWV, IVT, IVT_x, IVT_y,])
    new_ds = xr.merge(merge_data)

    return new_ds



if __name__ == "__main__": 


    test_file = "/expanse/lustre/scratch/t2hsu/temp_project/CW3E_WRF_RUNS/0.08deg/exp_Baseline01/runs/Baseline01_ens00/output/wrfout/wrfout_d01_2023-01-06_00:00:00_temp"
    output_file = "test_AR_diag.nc"

    print("Test file: %s" % (test_file,))
    ds_original = xr.open_dataset(test_file)
    
    print("Generating AR Diagnostics...")
    ds_diag = generateARDiag(ds_original)

    print("Writing output: ", output_file)
    ds_diag.to_netcdf(output_file)
