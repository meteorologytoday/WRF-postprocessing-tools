import xarray as xr
import pandas as pd
import numpy as np
import os
import os.path
import re
from datetime import datetime

engine = "netcdf4"
wrfout_time_fmt="%Y-%m-%d_%H:%M:%S"
wrfout_prefix="wrfout_d01_"
wrfout_suffix=""

class WRFSimMetadata:

    def __init__(self, start_datetime, data_interval, frames_per_file):

        self.start_datetime = pd.Timestamp(start_datetime)
        self.data_interval = data_interval
        self.frames_per_file = frames_per_file
        self.file_interval = frames_per_file * data_interval


def missingFiles(fnames):
    
    missing_files = []

    for fname in fnames:
        if not os.path.isfile(fname):
            missing_files.append(fname)

    return missing_files
        
def findfirst(a):

    idx = -1

    for i, _a in enumerate(a):
        if _a:
            idx = i
            break

    return idx

def findArgRange(arr, lb, ub, inclusive="both"):
    if lb > ub:
        raise Exception("Lower bound should be no larger than upper bound")

    if np.any( (arr[1:] - arr[:-1]) <= 0 ):
        raise Exception("input array should be monotonically increasing")

    if inclusive == "both":
        idx = np.logical_and((lb <= arr),  (arr <= ub))
    elif inclusive == "left":
        idx = np.logical_and((lb <= arr),  (arr < ub))
    elif inclusive == "right":
        idx = np.logical_and((lb < arr),  (arr <= ub))
    
    idx_low = findfirst(idx)
    idx_max = findlast(idx)

    return idx_low, idx_max




def computeIndex(wsm, index_time=None, time_passed=None):
    
    if not ( ( index_time is None ) ^ (time_passed is None) ):
        raise Exception("Only one of the parameters, `index_time` and `time_passed`, can and must be given.")


    if index_time is not None:
        if type(index_time) == str:
            index_time = pd.Timestamp(index_time) 
     
        delta_time = index_time - wsm.start_datetime
    
    elif time_passed is not None:
        delta_time = time_passed


    delta_frames = delta_time / wsm.data_interval

    if delta_frames % 1.0 != 0.0:

        print("Debug message: computed delta_time = ", delta_time)
        print("Debug message: computed delta_frames = ", delta_frames)

        if index_time is not None:
            raise Exception("The provided `index_time` is not a multiple of `data_interval` away from `start_datetime`.")
        elif time_passed is not None:
            raise Exception("The provided `time_passed` is not a multiple of `data_interval` away from `start_datetime`.")


    # Now compute the file and frame in file
    delta_frames = int(delta_frames)
    delta_files = int( np.floor( delta_frames / wsm.frames_per_file ) )
    frame = delta_frames - delta_files * wsm.frames_per_file

    file_time = wsm.start_datetime + delta_files * wsm.file_interval

    return file_time, frame


def genInclusiveBounds(wsm, beg_dt, end_dt, interval, inclusive):

    if inclusive == "left":
        _beg_dt = beg_dt
        _end_dt = end_dt - wsm.data_interval

    elif inclusive == "right":
        _beg_dt = beg_dt + wsm.data_interval
        _end_dt = end_dt

    elif inclusive == "both":
        _beg_dt = beg_dt
        _end_dt = end_dt

    elif inclusive == "neither":
        _beg_dt = beg_dt + wsm.data_interval
        _end_dt = end_dt - wsm.data_interval

    else:
        raise Exception("Unknown `inclusive` keyword %s. Only accept: left, right, both, neither." % (inclusive,))

    return _beg_dt, _end_dt


def genFilenameFromDateRange(wsm, time_rng, inclusive="left", prefix=wrfout_prefix, suffix="", dirname=None, time_fmt=wrfout_time_fmt):
   
    beg_dt, end_dt = genInclusiveBounds(wsm, time_rng[0], time_rng[1], wsm.data_interval, inclusive)
    print("beg_dt = ", beg_dt)   
    print("end_dt = ", end_dt)   
    print("wsm.data_interval = ", wsm.data_interval)   
     
    firstfile_dt, _ = computeIndex(wsm, index_time=beg_dt)
    lastfile_dt,  _ = computeIndex(wsm, index_time=end_dt)

    dts = pd.date_range(start=firstfile_dt, end=lastfile_dt, freq=wsm.file_interval, inclusive="both")
    
    fnames = [ "%s%s%s" % (prefix, dt.strftime(time_fmt), suffix, ) for dt in dts ]

    if dirname is not None:
        for i in range(len(fnames)):
            fnames[i] = os.path.join(dirname, fnames[i])
    
    return fnames


def listWRFOutputFiles(dirname, prefix=wrfout_prefix, suffix="", append_dirname=False, time_rng=None):

    valid_files = []
    
    pattern = "^%s(?P<DATETIME>[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}:[0-9]{2})$" % (prefix,)
    ptn = re.compile(pattern)
    file_times = []

    filter_time = False
    if time_rng is not None:
        filter_time = True


    for fname in os.listdir(dirname):
            
        m =  ptn.match(fname)

        if m:
            if append_dirname:
                fname = os.path.join(dirname, fname)

            if filter_time:
                t = pd.Timestamp(datetime.strptime(m.group('DATETIME'), "%Y-%m-%d_%H:%M:%S"))

                if time_rng[0] <= t and t < time_rng[1]:
                    valid_files.append(fname)
            else:
                valid_files.append(fname)

    valid_files.sort()


    return valid_files

 
def _loadWRFTimeOnly(filename):
   
    with xr.open_dataset(filename, engine=engine, decode_times=False,) as ds:
        t = [pd.Timestamp("%s %s" % (t[0:10], t[11:19])) for t in ds.Times.astype(str).to_numpy()]
    
    return t


def loadWRFDataFromDir(
    wsm,
    input_dir,
    beg_time,
    end_time=None,
    prefix=wrfout_prefix,
    suffix="",
    time_fmt=wrfout_time_fmt,
    verbose=False,
    avg=None,
    inclusive="left",
    drop_latlon_time_dependency=True,
):

    if drop_latlon_time_dependency:
        print("Warning: `drop_latlon_time_dependency` is set.")
    
    if end_time is None:
        inclusive = "both"
        end_time = beg_time
        print("Set inclusive = both")
  
    if verbose:
        print("[loadWRFDataFromDir] Going to load time range: [", beg_time, ", ", end_time, "]")
 
    fnames = genFilenameFromDateRange(
        wsm,
        [beg_time, end_time],
        inclusive=inclusive,
        prefix=prefix,
        suffix=suffix,
        dirname=input_dir,
        time_fmt=time_fmt,
    )
 
    if verbose:
        print("[loadWRFDataFromDir] Going to load files: ")
        for i, fname in enumerate(fnames):
            print("[%d] %s" % (i, fname,))
    
 
    # Check if all file exists
    missing_files = missingFiles(fnames)
    if len(missing_files) != 0:
        print("Error: some files do not exist.")
        for i, missing_file in enumerate(missing_files):
            print("Missing file %d : %s" % (i, missing_file,))

        raise Exception("Some files do not exist.")
 
    # Do a test to see if file contain 'time' (values) or 'Times' (String)
    test_ds = xr.open_dataset(fnames[0], decode_times=False, engine=engine)
  
    if 'time' in test_ds:
        print(fnames)
        ds = xr.open_mfdataset(fnames, decode_times=True, engine=engine, concat_dim=["time"], combine='nested')

 
    else:
        ds = xr.open_mfdataset(fnames, decode_times=False, engine=engine, concat_dim=["Time"], combine='nested')

        # ===== The following codes create a `time` coordinate with dtype = datetime64[ns] =====

        # pandas cannot handle 0001-01-01
        t = []
        
        for t_str in ds.Times.astype(str).to_numpy():

            YMD = t_str[0:10]
            HMS = t_str[11:19]

            #print("tstr = ", t_str)
            #print("YMD HMS = ", YMD, "; ", HMS)

            t.append(pd.Timestamp("%s %s" % (YMD, HMS)))

        ts = xr.DataArray(
            data = t,
            dims = ["Time"],
        ).rename('time_tmp').rename({"Time":'time'})
   
        ds = ds.rename_dims({"Time":'time'})
        ds = xr.merge([ds, ts]).rename({'time_tmp':'time'})
        ds = ds.set_coords("time")

        # ===== END =====

    # Select time
    ds_dts = ds.time.to_numpy()
    select_dts = pd.date_range(
        start     = beg_time, 
        end       = end_time,
        freq      = wsm.data_interval,
        inclusive = inclusive,
    )

    #print("Slected dts = ", select_dts )

    start_select = findfirst(ds_dts == beg_time)
    if start_select == -1:
        raise Exception("Error: Cannot find the matching `beg_time` = %s" % (str(beg_time),))
        
    for i, select_dt in enumerate(select_dts):
        if ds_dts[start_select + i] != select_dt:
            raise Exception("Error: Cannot find the matching `time` = %s" % (str(select_dt),))
            
    ds = ds.isel( time = slice(start_select, start_select + len(select_dts) + 1 ) )
    ts = ds.coords["time"] 
    if verbose:
        print("Loaded time: ")
        for i, _t in enumerate(pd.DatetimeIndex(ds.time)):
            print("[%d] %s" % (i, _t.strftime("%Y-%m-%d_%H:%M:%S")))


    has_xlat_xlong = np.all( [ coord_varname in ds.coords for coord_varname in ["XLAT", "XLONG"] ] )


    if drop_latlon_time_dependency and has_xlat_xlong:

        print("Dropping time dependency of XLAT and XLONG")

        test_coordnames = ['XLAT', 'XLONG', 'XLAT_U', 'XLAT_V', "XLONG_U", "XLONG_V"]
        coordnames = []

       
        # Need to detect if coordname is in the dataset.
        # If ds does not have this coordname, reset_coords will fail 
        for coordname in test_coordnames:
            if coordname in ds.coords:
                coordnames.append(coordname)

        ds = ds.reset_coords(names=coordnames)

        newly_assign_coords = {
            coordname : ds[coordname].isel(time=0, drop=True)
            for coordname in coordnames
        }
        ds = ds.drop_vars(coordnames)
        ds = ds.assign_coords(newly_assign_coords)

            
    print("WRF file has coordinate XLAT and XLONG? ", has_xlat_xlong)
    if avg is not None:

        # Unset XLAT and XLONG as coordinate
        # For some reason they disappeared after taking the time mean
        if has_xlat_xlong and (not drop_latlon_time_dependency):
            ds = ds.reset_coords(names=['XLAT', 'XLONG'])
        
        avg_all = avg == "ALL"

        if avg_all: 
            verbose and print("Averge ALL data")
            ds = ds.mean(dim="time", keep_attrs=True).expand_dims(dim={"time": ts[0:1]}, axis=0)
        
        elif type(avg) == pd.Timedelta:

            avg_N = avg / wsm.data_interval
            if avg_N % 1 != 0:
                raise Exception("Average time is not a multiple of wsm.data_interval.")
            else:
                avg_N = int(avg_N)
        
            groupped_time = np.zeros_like(ds.coords["time"])
            number_of_groups = ds.dims["time"] / avg_N
            if number_of_groups % 1 != 0:
                print("Warning (loadWRFDataFromDir): There are %d of data will not be used to produce averge." % 
                    (int(ds.dims["time"] - np.floor(number_of_groups)*avg_N),)
                )

            number_of_groups = int(np.floor(number_of_groups))

            verbose and print("avg_N = %d, number_of_groups = %d" % (avg_N, number_of_groups))

            for n in range(number_of_groups):
                groupped_time[(n*avg_N):((n+1)*avg_N+1)] = ds.coords["time"][n*avg_N].to_numpy()

            verbose and print(groupped_time)

            groupped_time = xr.DataArray(
                data=groupped_time,
                dims=["time"],
                coords=dict(time=ds.coords["time"])
            ).rename("groupped_time")

            ds = xr.merge([ds, groupped_time])
            ds = ds.groupby("groupped_time", squeeze=False).mean(dim="time").rename({"groupped_time": "time"})


        else:
            raise Exception("If avg is not None or string 'ALL', it has to be a pandas.Timedelta object. Now type(avg) = `%s`" % str(type(avg),) )


        if has_xlat_xlong and (not drop_latlon_time_dependency):
            ds = fixLostCoord(ds)

    return ds

def fixLostCoord(ds):

    print("##### Before Fix lost coord")
    print(ds)


    print("##### Fix lost coord")
    fixed_ds = ds.assign_coords(
        XLAT=( ds["XLAT"].dims, ds["XLAT"].data), 
        XLONG=( ds["XLONG"].dims, ds["XLONG"].data),
    )

    return fixed_ds

def loadWRFData(wsm, filename=None):
     
    ds = xr.open_dataset(filename, decode_times=False, engine=engine)
    t = [ pd.Timestamp("%s %s" % (t[0:10], t[11:19])) for t in ds.Times.astype(str).to_numpy() ]
    
    ts = xr.DataArray(
        data = t,
        dims = ["Time"],
    ).rename('time')
    
    ds = xr.merge([ds, ts]).rename({'Time':'time'})
    
    return ds
