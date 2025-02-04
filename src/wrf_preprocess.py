import xarray as xr
import pandas as pd
import numpy as np
from shared_constants import *
import scipy.signal

def isIn(pool, *xs):
    if pool is None:
        return True

    for x in xs:
    
   
        if x in pool:
            return True

    return False


def integrateVertically(X, ds, avg=False):

    MUB = ds.MUB
    DNW = ds.DNW
    MU_FULL = ds.MU + ds.MUB
    MU_STAR = MU_FULL / MUB
    integration_factor = - MUB * DNW / g0  # notice that DNW is negative, so we need to multiply by -1

    X_STAR = X * MU_STAR
    X_INT = (integration_factor * X_STAR).sum(dim="bottom_top")

    if avg:
        sum_integration_factor = integration_factor.sum(dim="bottom_top")
        X_INT = X_INT / sum_integration_factor

    return X_INT

def genKernel(size, shape, sigma=1):

    """
    Creates a 2D Gaussian kernel.

    Parameters:
        size: The size of the kernel (should be an odd integer).
        sigma: The standard deviation of the Gaussian distribution.
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    if shape == "gaussian":
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    elif shape == "flat":
        g = np.zeros_like(x) + 1
    else:
        raise Exception("Unknown shape: %s" % (shape,))

    return g / g.sum()


def filterVariable(da, kernel):

    t_len = len(da.coords["time"])
    da_lp = da.copy()
    for t in range(t_len):
        
        tmp = da.isel(time=t).to_numpy()
        tmp_lp = scipy.signal.convolve2d(tmp, kernel, mode='same', boundary='symm')
        
        da_lp[t, :, :] = tmp_lp


    return da_lp


"""

def genAnalysis(
    ds,
    data_interval,
):

    ref_ds = ds.mean(dim=['time'], keep_attrs=True)
    Nx = ref_ds.dims['west_east']
    Nz = ref_ds.dims['bottom_top']

    X_sU = ds.DX * np.arange(Nx+1) / 1e3
    X_sT = (X_sU[1:] + X_sU[:-1]) / 2
    X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)
    X_W = np.repeat(np.reshape(X_sT, (1, -1)), [Nz+1,], axis=0)
    dX_sT = ds.DX * np.arange(Nx)

    Z_W = (ref_ds.PHB + ref_ds.PH) / 9.81
    Z_T = (Z_W[1:, :] + Z_W[:-1, :]) / 2

    ds = ds.assign_coords(dict(
        west_east = X_sT, 
        west_east_stag = X_sU, 
    ))

    merge_data = []

    # Cannot use the following to get surface pressure:
    #PRES = ds.PB + ds.P
    #SFC_PRES = PRES.isel(bottom_top=0)
    
    # This is the correct one
    SFC_PRES = ds["PSFC"]

    PRES1000hPa=1e5

    R_over_cp = 2.0 / 7.0

    dT = (np.amax(ds["TSK"].to_numpy()) - np.amin(ds["TSK"].to_numpy())) / 2 

    TA = ( 300.0 + ds["T"].isel(bottom_top=0) ).rename("TA")
    TO = (ds["TSK"] * (PRES1000hPa/SFC_PRES)**R_over_cp).rename("TO")
    TOA    = ( TO - TA ).rename("TOA")

    #  e1=svp1*exp(svp2*(tgdsa(i)-svpt0)/(tgdsa(i)-svp3)) 


    # Bolton (1980). But the formula is read from 
    # phys/physics_mmm/sf_sfclayrev.F90 Lines 281-285 (WRFV4.6.0)
    salinity_factor = 0.98
    E1 = 0.6112e3 * np.exp(17.67 * (ds["TSK"] - 273.15) / (ds["TSK"] - 29.65) ) * salinity_factor
    QSFCMR = (287/461.6) * E1 / (SFC_PRES - E1)
    
    QA  = ds["QVAPOR"].isel(bottom_top=0).rename("QA")
    QO  = QSFCMR.rename("QO")

    QOA = QO - QA
    #QOA = xr.where(QOA > 0, QOA, 0.0)
    QOA = QOA.rename("QOA")
 
    #merge_data.append(WIND10)
    merge_data.extend([TO, TA, TOA, QO, QA, QOA,])

    V_T = ds["V"]
    
    _tmp = ds["U"]
    U_T = ds["T"].copy().rename("U_T")

    U_T[:, :, :] = (_tmp.isel(west_east_stag=slice(1, None)).to_numpy() + _tmp.isel(west_east_stag=slice(0, -1)).to_numpy()) / 2
    U_T = U_T.rename("U_T")
    merge_data.append(U_T) 
 
    U_sfc = U_T.isel(bottom_top=0).to_numpy()
    V_sfc = V_T.isel(bottom_top=0)
    WND_sfc = (U_sfc**2 + V_sfc**2)**0.5
    WND_sfc = WND_sfc.rename("WND_sfc")

    TTL_RAIN = ds["RAINNC"] + ds["RAINC"] #+ ds["RAINSH"] + ds["SNOWNC"] + ds["HAILNC"] + ds["GRAUPELNC"]
    PRECIP = ( TTL_RAIN - TTL_RAIN.shift(time=1) ) / data_interval.total_seconds() * 3600.0 # mm / hr
    
    TTL_RAIN = TTL_RAIN.rename("TTL_RAIN")
    PRECIP = PRECIP.rename("PRECIP") 


   
    #if "QICE_TTL" in ds:   
    #    WATER_TTL = ds["QVAPOR_TTL"] + ds["QRAIN_TTL"] + ds["QICE_TTL"] + ds["QSNOW_TTL"] + ds["QCLOUD_TTL"]
    #else:
    #    WATER_TTL = ds["QVAPOR_TTL"] + ds["QRAIN_TTL"] + ds["QCLOUD_TTL"]
    
    #dWATER_TTLdt = ( WATER_TTL - WATER_TTL.shift(time=1) ) / wrfout_data_interval.total_seconds()
    #dWATER_TTLdt = dWATER_TTLdt.rename("dWATER_TTLdt") 
    
    merge_data.append(PRECIP)
    merge_data.append(TTL_RAIN)
    #merge_data.append(dWATER_TTLdt)



    DIV10 = ( ( ds["U10"].roll(west_east=-1) - ds["U10"] ) / ds.DX ).rename("DIV10")
    VOR10 = ( ( ds["V10"].roll(west_east=-1) - ds["V10"] ) / ds.DX ).rename("VOR10")
    merge_data.append(DIV10)
    merge_data.append(VOR10)


    DIV = xr.zeros_like(ds["T"]).rename("DIV")
    tmp = ( ds["U"].roll(west_east_stag=-1) - ds["U"] ) / ds.DX
    tmp = tmp.isel(west_east_stag=slice(0, -1))
    DIV[:] = tmp[:]

    VOR = xr.zeros_like(ds["V"]).rename("VOR")
    tmp = ( ds["V"] - ds["V"].roll(west_east=1) ) / ds.DX
    tmp = (tmp.roll(west_east=-1) + tmp ) / 2.0
    VOR[:] = tmp[:]

    merge_data.append(DIV)
    merge_data.append(VOR)

    U_T = xr.zeros_like(ds["T"]).rename("U_T")
    tmp = (ds["U"].roll(west_east_stag=-1) + ds["U"] ) / 2.0
    tmp = tmp.isel(west_east_stag=slice(0, -1))
    U_T[:] = tmp[:]

    WND = ( U_T**2 + ds["V"]**2 )**0.5
    WND = WND.rename("WND")

    merge_data.append(U_T)
    merge_data.append(WND)

    print("PRECIP:", PRECIP)

    new_ds = xr.merge(merge_data)
    
    print(list(new_ds.keys())


    return new_ds

"""
def genAnalysis(
    ds,
    data_interval,
    varnames = None,
):
    
    merge_data = []


    if isIn(varnames, "PH850", "PH500"):

        ds_col = ds.mean(dim=["time", "south_north", "west_east"])
        base_pressure = ds_col.PB.to_numpy()
        ps = [850, 500] # hPa
        zidxs = []
        for p in ps:
            zidxs.append(
                np.argmin(np.abs(base_pressure - p*100))
            )

            print("Found constant pressure surface for p=%d hPa => zidx=%d. Exact number is: %f hPa" % (p, zidxs[-1], base_pressure[zidxs[-1]]/1e2) )


        PH_TTL = (ds.PHB + ds.PH).rename("PH_TTL")

        for var, prefix in [
            (PH_TTL, "PH"),
        ]:
            for i, p in enumerate(ps):
                newname = "%s%d" % (prefix, p,)
                zidx = zidxs[i]
                tmp = var.isel(bottom_top_stag = zidx)
                tmp = tmp.rename(newname)
                print("Add new variable: ", newname)
                merge_data.append(
                    tmp
                )

    if isIn(varnames, "TTL_RAIN", "PRECIP", "TTL_RAIN_LP", "RAINNC_LP", "RAINC_LP"):
        TTL_RAIN = ds["RAINNC"] + ds["RAINC"] #+ ds["RAINSH"] + ds["SNOWNC"] + ds["HAILNC"] + ds["GRAUPELNC"]
        TTL_RAIN = TTL_RAIN.rename("TTL_RAIN")
        merge_data.append(TTL_RAIN)

        PRECIP = ( TTL_RAIN - TTL_RAIN.shift(time=1) ) / (data_interval.total_seconds() / 3600.0) # mm / hr
        #print("PRECIP BEFORE: ", PRECIP)
        #PRECIP = PRECIP.shift(time=-1).rename(time="time_mid").isel(time_mid=slice(0, -1)).drop_vars("XTIME").rename("PRECIP") 
        PRECIP = PRECIP.shift(time=-1).rename(time="time_mid").isel(time_mid=slice(0, -1)).rename("PRECIP").drop_vars("XTIME", errors="ignore")

        #if "XTIME" in PRECIP.keys():
        #    PRECIP = PRECIP.drop_vars("XTIME")
 
        merge_data.append(PRECIP)

        #print("PRECIP: ", PRECIP)
        #print(np.any(np.isfinite(PRECIP.to_numpy())))
        # Low pass
        LP_kernel   = genKernel(5, "flat")
        TTL_RAIN_LP = filterVariable(TTL_RAIN, LP_kernel).rename("TTL_RAIN_LP")
        RAINNC_LP   = filterVariable(ds["RAINNC"], LP_kernel).rename("RAINNC_LP")
        RAINC_LP    = filterVariable(ds["RAINC"], LP_kernel).rename("RAINC_LP")
        merge_data.append(TTL_RAIN_LP)
        merge_data.append(RAINNC_LP)
        merge_data.append(RAINC_LP)
     

    if isIn(varnames, "SST_NOLND"):
       
        # LAND = 1, WATER = 2 
        SST_NOLND = ds["SST"].where(ds["SST"] != 0).rename("SST_NOLND")
        merge_data.append(SST_NOLND)

    
    if isIn(varnames, "IVT", "IVT_x", "IVT_y"):
        # Convert V_T
        V_T = xr.zeros_like(ds["T"]).rename("V_T")
        _tmp = ds["V"].to_numpy()
        V_T[:, :, :, :] = _tmp[:, :, 0:1, :]


        # Convert U_T
        _tmp = ds["U"].to_numpy()
        U_T = xr.zeros_like(ds["T"]).rename("U_T")

        U_T[:, :, :, :] = (_tmp[:, :, :, 1:] + _tmp[:, :, :, :-1]) / 2
        U_T = U_T.rename("U_T")
     
        # IWV
        IWV = integrateVertically(ds["QVAPOR"], ds, avg=False).rename("IWV")
     
        # IVT
        IVT_x = integrateVertically(ds["QVAPOR"] * U_T, ds, avg=False)
        IVT_y = integrateVertically(ds["QVAPOR"] * V_T, ds, avg=False)
        
        # code order matters here
        IVT = ((IVT_x**2 + IVT_y**2)**0.5).rename("IVT")
        IVT_x = IVT_x.rename("IVT_x")
        IVT_y = IVT_y.rename("IVT_y")

        merge_data.extend([IWV, IVT, IVT_x, IVT_y])




    new_ds = xr.merge(merge_data)
    return new_ds

