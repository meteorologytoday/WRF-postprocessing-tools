from multiprocessing import Pool
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

    try:    
        ds = xr.open_dataset(filename, engine="netcdf4")
    except Exception as e:
        
        print("Open file error. Conclude as all missing.")
    
        missing_varnames = [varname for varname in varnames]
        
        return missing_varnames

    ds_varnames = list(ds.keys())
    
    missing_varnames = []

    for varname in varnames:
        if varname not in ds_varnames:
            missing_varnames.append(varname)

    if len(missing_varnames) == 0:
        missing_varnames = None

    return missing_varnames





pressure_levs = [1000, 925, 850, 700, 600, 500, 400, 300, 200, 100, 50, 10]

def doWork(details):
    
    
    result = dict(details=details, status="UNKNOWN")

    input_file = Path(details["input_file"])
    output_file = Path(details["output_file"])
    check_policy = details["check_policy"]
    check_varnames = details["check_varnames"]
    overwrite = details["overwrite"]
    
    try:

        do_work = None

        if overwrite:

            do_work = True

        else:

            # Check if output file is good
            if check_policy == "exist":
                         
                if output_file.exists():
                    
                    do_work = False
                    
            elif check_policy == "varnames":        
                
                if output_file.exists():
                    missing_varnames = check(output_file, check_varnames) 

                    if missing_varnames is None:
                        
                       do_work = False

                    else:
                        
                        print("File %s is not okay after checking. Missing variables: %s" % (str(output_file), ", ".join(missing_varnames)))
                        do_work = True 
                else:
                    do_work = True 
            else:
                raise Exception("Error: check-policy %s does not exist." % (check_policy,))

                

        if do_work:
            
            output_dir = output_file.parents[0]
            output_dir.mkdir(parents=True, exist_ok=True)

            print("Input file: ", input_file)
            ds_original = xr.open_dataset(input_file, engine="netcdf4")
            ds_diag = generateReduction(ds_original)

            print("Writing output: ", output_file)
            ds_diag.to_netcdf(output_file)
                    
            print("Done. Now check it...")
            missing_varnames = check(output_file, check_varnames) 
            if missing_varnames is not None:
                raise Exception("Error: File %s is not okay after checking. Missing variables: %s" % (str(output_file), ", ".join(missing_varnames)))
            else:
                print("File %s is good." % (str(output_file),))

            # Explicitly close file IO.
            #ds_diag.close()
            #ds_original.close()

        else:
                    
            print("[Check-policy=%s] We do not have to do file %s." % (check_policy, str(output_file),))
            
            

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



if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process WRF file to reduce size.')
    parser.add_argument('--input', type=str, help='Input file or directory.', required=True)
    parser.add_argument('--output', type=str, help='Output file or directory.', required=True)
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--nproc', type=int, help='Number of tasks.', default=1)
    parser.add_argument('--number-of-files', type=int, help='Number of files in the directory to process. This is used when you just wanna do a few files', default=-1)
    
    parser.add_argument('--check-policy', type=str, help='Check policy. Allowed value: `exist`, `varnames`.', choices=["exist", "varnames"], default="exist")
    parser.add_argument('--check-varnames', type=str, nargs="*", help='If `--check-policy` is `varnames`, then this will be used.', default=["SST", "IVT", "PSFC", "PH"])

    args = parser.parse_args()

    print(args)

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("Processing ...")

    test_files = []

    if input_path.exists():

        if input_path.is_file():
            print("Input is a file.")
            test_files.append([args.input, args.output])
        
        elif input_path.is_dir():
            
            for i, input_file in enumerate(sorted(input_path.iterdir())):

                if i == args.number_of_files:
                    print("`--number-of-files` reached. Break.")
                    break
                
                output_file = output_path / input_file.name
                #if args.overwrite or (not output_file.exists()):
                test_files.append([input_file, output_file])

            
        else:
            raise Exception("Error: `input` = '%s' is not a file or directory." % (input_path,))
            
   
    else:
        raise Exception("Error: `input` = `%s` does not exist." % (input_path,))
 

    input_args = []
    for input_file, output_file in test_files:
                    
        print("Need to process file %s => %s" % (input_file, output_file,))
        
        input_args.append((dict(
            input_file = input_file,
            output_file = output_file,
            check_policy = args.check_policy,
            check_varnames = args.check_varnames,
            overwrite = args.overwrite,
        ),)) 

   
    failed_details = [] 
    with Pool(processes=args.nproc) as pool:

        results = pool.starmap(doWork, input_args)

        for i, result in enumerate(results):
            if result['status'] != 'OK':
                print('!!! Failed to generate output of date %s.' % (result['details']['input_file'],))
                failed_details.append(result['details'])


    print("Tasks finished.")

    print("Failed dates: ")
    for i, failed_detail in enumerate(failed_details):
        print("%d : %s" % (i+1, failed_detail['input_file'],))


    print("Done.")
