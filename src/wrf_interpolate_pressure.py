import xarray as xr
import numpy as np
import os
import os.path
import re
from datetime import datetime

default_pressure_levs = [1000, 925, 850, 700, 600, 500, 400, 300, 200, 100, 50, 10]


def interpolatePressure(ds, varnames, p):
   

    # Pressure must be monotincally increasing when interpolating
    ds = ds.isel(bottom_top=slice(None, None, -1))
 
    full_pres = (ds.P + ds.PB) * 0.01
   
    p = np.sort(np.array(p, dtype=np.float64))
    

    test_dim = ('Time', 'bottom_top', 'south_north', 'west_east', )
 
    if full_pres.dims != test_dim:
        raise Exception('Error: dimension has to be time, z, x, y on T-grid.')
   
    # Check variables
    for varname in varnames:
        da = ds[varname]

        if da.dims != test_dim:
            raise Exception("Error: %s's dimension has to be (time, z, y, x) on T-grid." % (varname,))
   

    new_data_vars = []

    Nt = da.sizes['Time']
    Nz = da.sizes['bottom_top']
    Nx = da.sizes['west_east']
    Ny = da.sizes['south_north']
    Np = len(p)

    rearr_pres = np.moveaxis(full_pres.to_numpy(), [0, 1, 2, 3], [3, 0, 1, 2]).reshape((Nz, -1))

    da_p = xr.DataArray(
        data=p,
        dims=["pressure",],
        coords=dict(
            pressure=(["pressure",], p),
        ),
    
        attrs=dict(
            description="Pressure.",
            units="hPa",
        ),
    )

    for varname in varnames:
    
        print("Interpolating variable `%s`" % (varname,))

        da = ds[varname] 

        original_data = da.to_numpy()
        rearr_data = np.moveaxis(original_data, [0, 1, 2, 3], [3, 0, 1, 2]).reshape((Nz, -1))

        new_data = np.zeros( (Nt*Np*Ny*Nx,), dtype=original_data.dtype ).reshape((Np, -1))
        
        for i in range(rearr_data.shape[1]):
            new_data[:, i] = np.interp(p, rearr_pres[:, i], rearr_data[:, i], left=np.nan, right=np.nan)
            
        
           
        # Transform the axis back 
        new_data = np.moveaxis(new_data.reshape((Np, Ny, Nx, Nt)), [3, 0, 1, 2], [0, 1, 2, 3])
        
        new_data_vars.append(xr.DataArray(
            data = new_data,
            dims = ["Time", "pressure", "south_north", "west_east"],
            coords = dict(
                XLAT = ds.coords["XLAT"],
                XLONG = ds.coords["XLONG"],
                XTIME = ds.coords["XTIME"],
                pressure = da_p,
            )
        ).rename(varname))
    
    
    new_ds = xr.merge(new_data_vars)
    
    return new_ds


def generatePressureDiag(ds, p = default_pressure_levs):

    U_U  = ds["U"].drop_vars(["XLAT_U", "XLONG_U"])
    V_V  = ds["V"].drop_vars(["XLAT_V", "XLONG_V"])
    W_W  = ds["W"]
    PH_W = ds["PH"] + ds["PHB"]


    U_T = (U_U + U_U.shift(west_east_stag=-1)) / 2.0
    U_T = U_T.isel(west_east_stag=slice(None, -1)).rename({'west_east_stag' : 'west_east'}).rename("U")


    V_T = (V_V + V_V.shift(south_north_stag=-1)) / 2.0
    V_T = V_T.isel(south_north_stag=slice(None, -1)).rename({'south_north_stag' : 'south_north'}).rename("V")

    W_T = (W_W + W_W.shift(bottom_top_stag=-1)) / 2.0
    W_T = W_T.isel(bottom_top_stag=slice(None, -1)).rename({'bottom_top_stag' : 'bottom_top'}).rename("W")

    PH_T = (PH_W + PH_W.shift(bottom_top_stag=-1)) / 2.0
    PH_T = PH_T.isel(bottom_top_stag=slice(None, -1)).rename({'bottom_top_stag' : 'bottom_top'}).rename("PH")


    # Rotate velocity to zonal and meridional
    Uzl_T = U_T * ds["COSALPHA"] - V_T * ds["SINALPHA"]
    Vml_T = V_T * ds["COSALPHA"] + U_T * ds["SINALPHA"]

    Uzl_T = Uzl_T.rename("Uzl")
    Vml_T = Vml_T.rename("Vml")
    
    # Vorticity and Divergence
    VORT_Upart = - (U_T.shift(south_north=-1) - U_T.shift(south_north=1)) * ds["RDY"] / 2
    VORT_Vpart =   (V_T.shift(west_east=-1)   - V_T.shift(west_east=1))   * ds["RDX"] / 2
    VORT_T = ( VORT_Upart +  VORT_Vpart ).rename("VORT")

    DIV_Upart = (U_U - U_U.shift(west_east_stag=1)) * ds["RDX"]
    DIV_Upart = DIV_Upart.isel(west_east_stag=slice(1, None)).rename({'west_east_stag' : 'west_east'})
    DIV_Vpart = (V_V - V_V.shift(south_north_stag=1)) * ds["RDY"]
    DIV_Vpart = DIV_Vpart.isel(south_north_stag=slice(1, None)).rename({'south_north_stag' : 'south_north'})
    DIV_T = ( DIV_Upart +  DIV_Vpart ).rename("DIV")



 
    merged_ds = [Uzl_T, Vml_T, W_T, PH_T, VORT_T, DIV_T]

    for varname in ["T", "P", "PB", "QVAPOR", ]:
        merged_ds.append(ds[varname])
   
    merged_ds = xr.merge(merged_ds)
 
    new_ds = interpolatePressure(merged_ds, ["Uzl", "Vml", "VORT", "DIV", "W",  "PH", "QVAPOR"], p=p)
    
    return new_ds 
 

if __name__ == "__main__": 


    test_file = "/expanse/lustre/scratch/t2hsu/temp_project/CW3E_WRF_RUNS/0.08deg/exp_Baseline01/runs/Baseline01_ens00/output/wrfout/wrfout_d01_2023-01-06_00:00:00_temp"

    print("Test file: %s" % (test_file,))
    ds_original = xr.open_dataset(test_file)
    
    print("Generating Pressure Diagnostics...")
    ds_diag = generatePressureDiag(ds_original)


    ds_diag.to_netcdf("test_diag.nc")
