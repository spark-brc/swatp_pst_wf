from urllib.error import HTTPError
import time
import geopandas as gpd
import os
import numpy as np
import wget
import pandas as pd
import netCDF4 as nc
from termcolor import colored
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import psutil
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
from swatp_pst import handler, analyzer
import scipy.stats as scs
import spei as si

class ClimateDataDownloader:
    def __init__(self, working_dir, dataset_name, model_name, ssp_of_interest,
                 meta_data_format, variables_of_interest, versions_avail):
        self.working_dir = working_dir
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.ssp_of_interest = ssp_of_interest
        self.meta_data_format = meta_data_format
        self.variables_of_interest = variables_of_interest
        self.versions_avail = versions_avail
        self.west, self.south, self.east, self.north = self._get_bounds()
        self.dates_historical = np.arange(1950, 2015) # historical years from 1950 ~ 2014
        self.dates_projected = np.arange(2015, 2101) # projected years from 2015 ~ 2100
        self.max_retries = 3
        self.timeout = 10
        self.nworkers = self._core_workers()

    def _core_workers(self):
        return psutil.cpu_count(logical=False)

    def _get_bounds(self):
        for file in os.listdir(self.working_dir):
            if file.endswith(".shp"):
                shp_file_path = os.path.join(self.working_dir, file)
                gdf = gpd.read_file(shp_file_path)
                gdf = gdf.to_crs(epsg=4326) # Set the CRS to WGS84
                return gdf.total_bounds

    def download_nc_file(self, vers, var, ssp, date, save_folder):
        filename = f"{var}_day_{self.model_name}_{ssp}_{self.meta_data_format}_gn_{str(date)}{vers}.nc"
        save_path = os.path.join(save_folder, filename)

        if not os.path.exists(save_path):
            
            #NOTE: change start time T12:00:00Z -> T00:00:00Z to get 365 days
            wget_string = (
                f"https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/{self.dataset_name}/{self.model_name}/{ssp}/"
                f"{self.meta_data_format}/{var}/{filename}?var={var}&north={self.north}&west={self.west}&east={self.east}&south={self.south}"
                f"&horizStride=1&time_start={date}-01-01T00:00:00Z"
                f"&time_end={date}-12-26T12:00:00Z&&&accept=netcdf3&addLatLon=true"
            )
            print(wget_string)
            wget.download(wget_string, save_path, bar=None)

    def download_all_single(self):
        for ssp in self.ssp_of_interest:
            dates = self.dates_historical if ssp == "historical" else self.dates_projected

            for var in self.variables_of_interest:
                save_folder = f'{self.working_dir}/{self.dataset_name}/{self.model_name}/{ssp}/{self.meta_data_format}/{var}'
                os.makedirs(save_folder, exist_ok=True)

                for date in dates:
                    vers = self.versions_avail[-1]  # Start with the last version
                    for attempt in range(self.max_retries):
                        try:
                            self.download_nc_file(vers, var, ssp, date, save_folder)
                            print(f"Download Successful for Dataset: {self.dataset_name}, Model: {self.model_name}, "
                                  f"ssp: {ssp}, Variable: {var}, Version: {vers}, Date:{date}")
                            break
                        except HTTPError as e:
                            if e.code == 504 and attempt < self.max_retries - 1:
                                print(
                                    f"Gateway Timeout (504) on attempt {attempt + 1} for version {vers}. Retrying in {self.timeout} seconds...")
                                time.sleep(self.timeout)
                            else:
                                # Try the next version
                                if vers == self.versions_avail[2]:
                                    vers = self.versions_avail[1]
                                elif vers == self.versions_avail[1]:
                                    vers = self.versions_avail[0]
                                else:
                                    print(f"All versions failed to download for {var} on {date}.")
                                    break

    def _conv_360_180(self, lon):
        newlon = (float(lon) + 180) % 360 - 180
        return f"{newlon:.3f}"


    def download_all(self):
        time = datetime.now().strftime('- %m/%d/%y %H:%M:%S -')
        if os.path.exists(os.path.join(self.working_dir, "downloadednc.log")):
            os.remove(os.path.join(self.working_dir, "downloadednc.log"))
        with open(os.path.join(self.working_dir, "downloadednc.log"), "w") as f:
            f.write(f"# log files created by nc2swat ... {time}\n")

        print(
            f"\n > Start downloading dataset in parallel processing"
            + colored(f" with {self.nworkers} workers", 'magenta')
            + colored(
                " ... initiated", 'blue') + "\n"
                f"   D: Dataset  | M: Model  | ssp: SSP  | Var: Variable  | Ver: Version  | Date: Date   \n")
        
        tasks = []
        with ThreadPoolExecutor(max_workers=self.nworkers) as executor:  # Adjust max_workers as needed and use number of cores
            for ssp in self.ssp_of_interest:
                dates = self.dates_historical if ssp == "historical" else self.dates_projected

                for var in self.variables_of_interest:
                    save_folder = f'{self.working_dir}/{self.dataset_name}/{self.model_name}/{ssp}/{self.meta_data_format}/{var}'
                    os.makedirs(save_folder, exist_ok=True)

                    for date in dates:
                            vers = self.versions_avail[-1]  # Start with the last version
                            tasks.append(
                                executor.submit(self.download_with_retries, vers, var, ssp, date, save_folder)
                            )

            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as e: # Raise any exceptions that occurred during download
                    print(f"Download failed with exception: {e}")

    def download_with_retries(self, vers, var, ssp, date, save_folder):
        for attempt in range(self.max_retries):
            try:
                self.download_nc_file(vers, var, ssp, date, save_folder)
                print(f"  ... D: {self.dataset_name}, M: {self.model_name}, " +
                      f"ssp: {ssp}, Var: {var}, Ver: {vers}, Date:{date}" + colored(" ... OK", 'green'))
                self.write_ok_log_file(ssp, var, date)
                return
            except HTTPError as e:
                if e.code == 504 and attempt < self.max_retries - 1:
                    print(
                        f"  ... Gateway Timeout (504) on attempt {attempt + 1} for version {vers}. Retrying in {self.timeout} seconds...")
                    time.sleep(self.timeout)
                else:
                    # Try the next version
                    if vers == self.versions_avail[2]:
                        vers = self.versions_avail[1]
                    elif vers == self.versions_avail[1]:
                        vers = self.versions_avail[0]
                    else:
                        print(f"  ... All versions for {ssp} failed to download for {var} on {date}" + colored(" ... failed", 'red'))
                        self.write_failed_log_file(ssp, var, date)
                        return

                    
    def process_netcdf(self):
        data_dir = os.path.join(self.working_dir, f"{self.dataset_name}/{self.model_name}")
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        data_dict_hist = {}
        data_dict_ssp = {}
        for ssp in sorted(os.listdir(data_dir)):
            metadata_pth = os.path.join(data_dir, ssp)
            data_dict_proj = {}
            for metadata_info in sorted(os.listdir(metadata_pth)):
                var_path = os.path.join(metadata_pth, metadata_info)

                for var in sorted(os.listdir(var_path)):
                    climate_info_path = os.path.join(var_path, var)
                    df = pd.DataFrame()
                    first_time = True
                    for data_file_nm in sorted(os.listdir(climate_info_path)):
                        # print(data_file_nm)
                        data_file_pth = os.path.join(climate_info_path, data_file_nm)
                        data_file = nc.Dataset(data_file_pth, mode='r')
                        num_days = data_file[var].shape[0]
                        var_data = data_file[var][:].reshape(num_days, -1)
                        latitudes = data_file.variables['lat'][:]
                        longitudes = data_file.variables['lon'][:]

                        # Get the time variable
                        time_var = data_file.variables['time']
                        # Get the time units and calendar attributes
                        time_units = time_var.units
                        calendar = time_var.calendar if hasattr(time_var, 'calendar') else 'standard'# convert time

                        # Convert time values to datetime objects
                        time_dates = nc.num2date(time_var[:], units=time_units, calendar=calendar)

                        # Create a list of lat-lon pair names for the columns
                        lat_lon_str = []
                        lat_lon_pairs = []
                        for lat in latitudes:
                            for lon in longitudes:
                                lat_lon_str.append(
                                    f"{int(lat * 1000)}_{int(float(self._conv_360_180(lon)) * 1000)}"
                                    )
                                lat_lon_pairs.append([f"{float(lat):.3f}", f"{float(self._conv_360_180(lon)):.3f}"])
                        if first_time:
                            df = pd.DataFrame(var_data, columns=lat_lon_str)
                            df['time'] = time_dates
                            df.set_index('time', inplace=True)
                            first_time = False
                        else:
                            temp_df = pd.DataFrame(var_data, columns=lat_lon_str)
                            temp_df['time'] = time_dates
                            temp_df.set_index('time', inplace=True)
                            df = pd.concat([df, temp_df])
                        data_file.close()

                    if ssp == "historical":
                        data_dict_hist[var] = df
                    else:
                        data_dict_proj[var] = df
            if not ssp == "historical":
                data_dict_ssp[ssp] = data_dict_proj
        return data_dict_ssp, data_dict_hist, lat_lon_pairs

    def convert_to_swat(self):
        print(
            f"\n > Start converting netcdf to SWAT weather input formats"
            + colored(
                " ... initiated", 'blue') + "\n")

        data_dict_ssp, data_dict_hist, lat_lon_pairs = self.process_netcdf()
        for ssp, dicts in data_dict_ssp.items():
            save_folder = f'{self.working_dir}/{self.model_name}_SWAT_files'
            os.makedirs(save_folder, exist_ok=True)

            projected_dict = dicts
            tasmax_df = pd.DataFrame()
            tasmin_df = pd.DataFrame()
            for var, hist_df in data_dict_hist.items():
                names = hist_df.columns.tolist()

                names_temp = ["temp_max_min" + "_" + item for item in names]
                names = [var + "_" + item for item in names]
                if var == "tasmin" or var == "tasmax":
                    names = names_temp

                ids = list(range(1, len(names) + 1))
                elevation = [100] * len(names)
                df_info = pd.DataFrame({
                    'ID': ids,
                    'NAME': names,
                    'LAT': [lat for lat, lon in lat_lon_pairs],
                    'LONG': [self._conv_360_180(lon) for lat, lon in lat_lon_pairs],
                    'ELEVATION': elevation
                })

                if var == "tasmin" or var == "tasmax":
                    save_path = os.path.join(save_folder, "tmp" + ".txt")
                else:
                    save_path = os.path.join(save_folder, self._change_clifilenam(var) + ".txt")
                df_info.to_csv(save_path, index=False) # write 
            
                projected_df = projected_dict[var]
                df_data = pd.concat([hist_df, projected_df])
                df_data.index = pd.to_datetime(df_data.index.astype(str))
                full_date_range = pd.date_range(start=df_data.index.min(), end=df_data.index.max(), freq='D')
                missing_dates = full_date_range.difference(df_data.index)
                for date in missing_dates:
                    print(
                        f" ... replaced missing value for {ssp}, {var}, {date.date()} with -99.0"
                        + colored(" ... missing", 'yellow'))
                df_data = df_data.reindex(full_date_range, fill_value=np.nan)
                # print(type(df_data))
                if var in ["tas", "tasmax", "tasmin"]:
                    df_data = df_data - 273.15  # convert to degrees celsius
                    
                if var == "pr":
                    df_data = df_data*86400  #1 kg/m2/s = 86400 mm/day
                df_data = df_data.round(3)
                data_save_pth = os.path.join(save_folder, ssp)
                os.makedirs(data_save_pth, exist_ok=True)
                date_string = df_data.index.min().strftime("%Y%m%d")

                if var == "tasmax":
                    tasmax_df = df_data
                elif var == "tasmin":
                    tasmin_df = df_data
                else:
                    if var in ["rlds", "rsds"]:
                        df_data = df_data * 0.0036  # Convert to MJ/m^2

                    for i, column in enumerate(df_data.columns):
                        
                        file_path = f'{data_save_pth}/{names[i]}.txt'
                        with open(file_path, 'w') as f:
                            f.write(date_string + '\n')  # Write the date as the first line
                            df_data.fillna(-99.0, inplace=True)
                            df_data[column].to_csv(f, index=False,
                                                   header=False, lineterminator='\n')  # Write the DataFrame column to the file

            for i, column in enumerate(tasmax_df.columns):
                # Create a combined DataFrame for each corresponding pair of columns
                combined_df = pd.DataFrame({
                    'tmax': tasmax_df[column],
                    'tmin': tasmin_df[column]
                })
                file_path = f'{data_save_pth}/{names[i]}.txt'
                with open(file_path, 'w') as f:
                    f.write(date_string + '\n')  # Write the date as the first line
                    combined_df.fillna(-99.0, inplace=True)
                    combined_df.to_csv(f, index=False, header=False, lineterminator='\n')  # Write the DataFrame column to the file

    def convert_to_swatplus(self):
        data_dict_ssp, data_dict_hist, lat_lon_pairs = self.process_netcdf()
        for ssp, dicts in data_dict_ssp.items():
            save_folder = f'{self.working_dir}/{self.model_name}_SWATplus_files'
            os.makedirs(save_folder, exist_ok=True)

            projected_dict = dicts
            tasmax_df = pd.DataFrame()
            tasmin_df = pd.DataFrame()
            for var, hist_df in data_dict_hist.items():
                names = hist_df.columns.tolist()

                names_temp = ["temp_max_min" + "_" + item for item in names]
                names = [var + "_" + item for item in names]
                if var == "tasmin" or var == "tasmax": 
                    names = names_temp

                ids = list(range(1, len(names) + 1))
                elevation = [100] * len(names)
                df_info = pd.DataFrame({
                    'ID': ids,
                    'NAME': names,
                    'LAT': [lat for lat, lon in lat_lon_pairs],
                    'LONG': [self._conv_360_180(lon) for lat, lon in lat_lon_pairs],
                    'ELEVATION': elevation
                })
                if var == "tasmin" or var == "tasmax":
                    filenam = "tmp.cli"
                    des = "temperature file names"
                    save_path = os.path.join(save_folder, filenam)
                elif var == "pr":
                    filenam = "pcp.cli"
                    des = "precipitation file names"
                    save_path = os.path.join(save_folder, filenam)
                elif var == "hurs":
                    filenam = "hmd.cli"
                    des = "humidity file names"
                    save_path = os.path.join(save_folder, filenam)
                elif var == "sfcWind":
                    filenam = "wnd.cli"
                    des = "wind speed file names"                    
                    save_path = os.path.join(save_folder, filenam)
                elif var == "rlds": #NOTE: check whether long or short
                    filenam = "slr.cli"
                    des = "solar radiation file names"        
                    save_path = os.path.join(save_folder, filenam)
                else:
                    save_path = os.path.join(save_folder, var + ".txt")

                firstline, secondline = self._header(filenam, des)
                with open(save_path, 'w') as f:
                    f.write(firstline + '\n')  # Write the date as the first line
                    f.write(secondline + '\n')  # Write the date as the first line
                    df_info['NAME'].to_csv(f, index=False,
                                            header=False, lineterminator='\n')  # Write the DataFrame column to the file
    '''
                # df_info.to_csv(save_path, index=False) # write climate info

                projected_df = projected_dict[var]
                df_data = pd.concat([hist_df, projected_df])

                df_data.index = pd.to_datetime(df_data.index.astype(str))
                full_date_range = pd.date_range(start=df_data.index.min(), end=df_data.index.max(), freq='D')
                missing_dates = full_date_range.difference(df_data.index)
                # for date in missing_dates:
                #     print(f"Missing dates for ssp: {ssp}, variable: {var}: {date.date()}")

                df_data = df_data.reindex(full_date_range, fill_value=-99.0)
                if var in ["tas", "tasmax", "tasmin"]:
                    df_data = df_data - 273.15  # convert to degrees celsius

                df_data = df_data.round(3)
                data_save_pth = os.path.join(save_folder, ssp)
                os.makedirs(data_save_pth, exist_ok=True)
                date_string = df_data.index.min().strftime("%Y%m%d")

                if var == "tasmax":
                    tasmax_df = df_data
                elif var == "tasmin":
                    tasmin_df = df_data
                else:
                    if var in ["rlds", "rsds"]:
                        df_data = df_data * 0.0036  # Convert to MJ/m^2


                    for i, column in enumerate(df_data.columns):
                        file_path = f'{data_save_pth}/{names[i]}.txt'
                        with open(file_path, 'w') as f:
                            f.write(date_string + '\n')  # Write the date as the first line
                            df_data[column].to_csv(f, index=False,
                                                   header=False, lineterminator='\n')  # Write the DataFrame column to the file

            for i, column in enumerate(tasmax_df.columns):
                # Create a combined DataFrame for each corresponding pair of columns
                combined_df = pd.DataFrame({
                    'tmax': tasmax_df[column],
                    'tmin': tasmin_df[column]
                })

                file_path = f'{data_save_pth}/{names[i]}.txt'
                with open(file_path, 'w') as f:
                    f.write(date_string + '\n')  # Write the date as the first line
                    combined_df.to_csv(f, index=False, header=False)  # Write the DataFrame column to the file
    '''

    def _header(self, filenam, des):
        time = datetime.now().strftime('- %m/%d/%y %H:%M:%S -')
        firstline = f"{filenam}: {des} - file written by ... {time}"
        secondline = "filename"
        return firstline, secondline

    def _change_clifilenam(self, clfnam):
        """Changes climate variable names to SWAT-compatible names.

        :param clfnam: Original climate variable name.
        :type clfnam: str
        :return: SWAT-compatible climate variable name.
        :rtype: str
        """

        mapping = {
            "hurs": "rh",
            "huss": "huss",
            "pr": "pcp",
            "rlds": "solar",
            "rsds": "rsds",
            "sfcWind": "wind",
            "tas": "tas",
            "tasmax": "tasmax",
            "tasmin": "tasmin"
        }

        return mapping.get(clfnam, "variable name doesn't exist")


    def write_ok_log_file(self, ssp, var, date):
        with open(os.path.join(self.working_dir, "downloadednc.log"), "a") as f:
            # f.write("# log files created by swatp_pst\n")
            f.write(f"{ssp},{var},{date},ok\n")

    def write_failed_log_file(self, ssp, var, date):
        with open(os.path.join(self.working_dir, "downloadednc.log"), "a") as f:
            # f.write("# log files created by swatp_pst\n")
            f.write(f"{ssp},{var},{date},failed\n")

def cli_down():
    working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\climate_scenarios\\test"
    # working_dir = "D:\\Projects\\Watersheds\\Mun\\climate_scenarios"
    dataset_name = "GDDP-CMIP6"
    model_name = "UKESM1-0-LL"
    ssp_of_interest = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
    meta_data_format = "r1i1p1f2"
    variables_of_interest = [
                    "hurs", "huss", "pr", "rlds", "rsds", 
                    "sfcWind", "tas", "tasmax", "tasmin"
                    ]
    versions_avail = ["", "_v1.1", "_v1.2"]

    # Instantiate the object
    downloader = ClimateDataDownloader(working_dir, dataset_name, model_name,
                                        ssp_of_interest, meta_data_format,
                                        variables_of_interest, versions_avail)

    # Download netcdf files
    downloader.download_all()

    # Process netcdf and convert to SWAT format
    # downloader.convert_to_swat()


class ClimateAnalyzer:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.scns, self.full_paths = self.get_weather_folder_lists()

    def get_weather_folder_lists(self):
        os.chdir(self.working_dir)
        scns = [name for name in os.listdir(".") if os.path.isdir(name)]
        full_paths = [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]
        return scns, full_paths
    
    def read_pcps(self, path, filenam):
        with open(os.path.join(path, filenam), "r") as tf:
            stdate = tf.readlines()[0]
        df = pd.read_csv(
                    os.path.join(path, filenam), skiprows=1, header=None, names=["pcp"]
                    )
        df.replace(-99, np.nan, inplace=True)
        df.index = pd.date_range(start=stdate, periods=len(df))
        return df

    def get_pcps(self):
        # find file start with pr
        tot_pcp = pd.DataFrame()
        for path, scnam in zip(self.full_paths, self.scns):
            files = [i for i in os.listdir(path) if i.startswith('pr')]            
            df_pcp = pd.DataFrame()
            for filenam in files:
                df_pcp = pd.concat([df_pcp, self.read_pcps(path, filenam)["pcp"]], axis=1)
            dfpcp_mean = df_pcp.mean(axis=1)
            dfpcp_mean.name = f"{scnam}_pcp"
            tot_pcp = pd.concat([tot_pcp, dfpcp_mean], axis=1)
        tot_pcp.index = pd.DatetimeIndex(tot_pcp.index).normalize()
        return tot_pcp

    def get_temperatures(self):
        # find file start with temp
        tot_max = pd.DataFrame()
        tot_min = pd.DataFrame()

        for path, scnam in zip(self.full_paths, self.scns):
            files = [i for i in os.listdir(path) if i.startswith('temp')]            
            df_max = pd.DataFrame()
            df_min = pd.DataFrame()
            for filenam in files:
                df_max = pd.concat([df_max, self.read_temps(path, filenam)["tmax"]], axis=1)
                df_min = pd.concat([df_min, self.read_temps(path, filenam)["tmin"]], axis=1)
            dfmax_mean = df_max.mean(axis=1)
            dfmin_mean = df_min.mean(axis=1)
            dfmax_mean.name = f"{scnam}_tmax"
            dfmin_mean.name = f"{scnam}_tmin"
            tot_max = pd.concat([tot_max, dfmax_mean], axis=1)
            tot_min = pd.concat([tot_min, dfmin_mean], axis=1)
        tot_min.index = pd.DatetimeIndex(tot_min.index).normalize()
        tot_max.index = pd.DatetimeIndex(tot_max.index).normalize()
        tot_mean = tot_min.copy()
        tot_mean.columns = tot_max.columns
        tot_mean = (tot_max + tot_mean)/2
        tot_mean.columns = [f"{col[:-5]}_tmean" for col in tot_max.columns]
        tot_mean.index = pd.DatetimeIndex(tot_mean.index).normalize()
        return tot_max, tot_min, tot_mean

    def read_temps(self, path, filenam):
        with open(os.path.join(path, filenam), "r") as tf:
            stdate = tf.readlines()[0]
        df = pd.read_csv(
                    os.path.join(path, filenam), skiprows=1, header=None, names=["tmax", "tmin"]
                    )
        df.replace(-99, np.nan, inplace=True)
        df.index = pd.date_range(start=stdate, periods=len(df))
        return df

    def plot_pcps(self):
        df = self.get_pcps()
        df = df[["ssp245_pcp", "ssp585_pcp"]]
        base = df['1/1/1980':'12/31/2014'].loc[:, "ssp245_pcp"]
        base.name = "Historical"
        nf = df['1/1/2015':'12/31/2040'].loc[:, ["ssp245_pcp", "ssp585_pcp"]]
        nf.columns = ["ssp245_nf", "ssp585_nf"]
        mf = df['1/1/2041':'12/31/2070'].loc[:, ["ssp245_pcp", "ssp585_pcp"]]
        mf.columns = ["ssp245_mf", "ssp585_mf"]
        ff = df['1/1/2071':'12/31/2099'].loc[:, ["ssp245_pcp", "ssp585_pcp"]]
        ff.columns = ["ssp245_ff", "ssp585_ff"]
        dff = pd.concat([base, nf, mf, ff], axis=1)
        xlabels = ["Historical" if x == "base" else x for x in dff.columns]


        # Boxplot
        f, axes = plt.subplots(2, 6, figsize=(16,6), sharex=True, sharey=True)
        ax1 = f.add_subplot(111, frameon=False)
        ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
        for i, ax in enumerate(axes.flat):
            df_m = dff.loc[dff.index.month==i+1]
            data = [df_m[i].dropna() for i in df_m.columns]
            # df_m = df_m.dropna(ax)

            # ax.boxplot(data, flierprops=flierprops)
            r = ax.violinplot(
                data,  widths=0.7, showmeans=True, showextrema=True,
                # bw_method='silverman'
                )
            r['cmeans'].set_color('r')
            ax.set_xticks([i+1 for i in range(len(xlabels))])
            ax.set_xticklabels(xlabels, rotation=90)
            # ax.set_xticks(df_m.columns[::1])
            ax.set_title(
                month_names[i],
                horizontalalignment='center',
                x=0.5,
                y=0.85,
                fontsize=12
            )
            ax.tick_params(axis='both', labelsize=12)
        ax1.set_ylabel('{}'.format('Monthly Rainfall Intensity $(mm/month)$'), fontsize=12, labelpad=10)
        # ax1.set_ylabel('Average Monthly Stream Discharge $(m^3/s)$', fontsize=12, labelpad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.working_dir, 'projected_viloin_mstr.png'), dpi=300, bbox_inches="tight")
        plt.show()

    def customize_cap_lengths(self, r, ax):
        # NOTE: customize cap lengths
        factor_x, factor_y = 1, 1 # factor to reduce the lengths
        for vp_part in ("cbars", "cmaxes", "cmins"):
            vp = r[vp_part]
            if vp_part in ("cmaxes", "cmins"):
                lines = vp.get_segments()
                new_lines = []
                for line in lines:
                    center = line.mean(axis=0)
                    line = (line - center) * np.array([factor_x, factor_y]) + center
                    new_lines.append(line)
                vp.set_segments(new_lines)
            vp.set_edgecolor("black")
        # ax.boxplot(df_m.values, flierprops=flierprops)

    def plot_temps(self):
        tot_max, tot_min, tot_mean = self.get_temperatures()
        tot_max = tot_max[["ssp245_tmax", "ssp585_tmax"]]
        tot_mean =tot_mean[["ssp245_tmean", "ssp585_tmean"]]
        tot_min =tot_min[["ssp245_tmin", "ssp585_tmin"]]
        adf_mean = tot_mean["1/1/1980":"12/31/2099"].resample('YE').mean()
        adf_min = tot_min["1/1/1980":"12/31/2099"].resample('YE').mean()
        adf_max = tot_max["1/1/1980":"12/31/2099"].resample('YE').mean()

        f, axes = plt.subplots(
            1, 4, figsize=(20, 9), sharey=True,
            gridspec_kw={
                        'width_ratios': [0.7, 0.04, 0.22, 0.04],
                        'wspace': 0.01
                        })

        ax1 = f.add_subplot(111, frameon=False)
        ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        axes[1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        axes[1].axis('off')

        # axes[1].axis('off')
        axes[3].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        axes[3].axis('off')

        ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        marker = itertools.cycle((',', '+', '.', 'o', '*', 'v', '^', '<', '>', 'D'))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i in range(len(adf_mean.columns)):
            axes[0].plot(adf_mean.index, adf_mean.iloc[:, i], marker = next(marker), label=adf_mean.columns[i])
            axes[0].plot(adf_mean.index, adf_min.iloc[:, i], color=colors[i], alpha=0.5)
            axes[0].fill_between(adf_mean.index, adf_min.iloc[:, i], adf_max.iloc[:, i], alpha=0.3)
            axes[0].plot(adf_mean.index, adf_max.iloc[:, i], color=colors[i], alpha=0.5)
        axes[0].plot(
            adf_mean["1/1/1980":"12/31/2014"].index, 
            adf_mean["1/1/1980":"12/31/2014"].loc[:, "ssp245_tmean"], 
            color="gray", zorder=10, linewidth=5, label="Historical Temperature")
            # ax.fill_between(scn_df.index, s585_min, s585_max, alpha=0.3, label='ssp585')

        t245_min = adf_min.loc[:, "ssp245_tmin"]
        t245_max = adf_max.loc[:, "ssp245_tmax"]
        t585_min = adf_min.loc[:, "ssp585_tmin"]
        t585_max = adf_max.loc[:, "ssp585_tmax"]

        # '''
        axes[2].fill_between(adf_mean.index, t245_min, t245_max, alpha=0.3, label='ssp245 ')
        axes[2].fill_between(adf_mean.index, t585_min, t585_max, alpha=0.3, label='ssp585')

        ntmax245 = t245_max['1980-1-1':'2019-12-31'].mean()
        ntmin245 = t245_min['1980-1-1':'2019-12-31'].mean()

        ntmax585 = t585_max['1980-1-1':'2019-12-31'].mean()
        ntmin585 = t585_min['1980-1-1':'2019-12-31'].mean()


        axes[1].bar(
            0, ntmax245-ntmin245,
            width=0.7,
            bottom=ntmin245, alpha=0.3)
        axes[1].bar(
            1, ntmax585-ntmin585,
            width=0.7,
            bottom=ntmin585, alpha=0.3)

        axes[1].text(
            0, (ntmax245+ntmin245)/2,
        'ssp245',
            rotation=90,
            va='top',
            ha='center',
            fontsize=12)
        axes[1].text(
            1, (ntmax585+ntmin585)/2,
            'ssp585',
            rotation=90,
            va='top',
            ha='center',
            fontsize=12
            )

        axes[1].text(
            0, ntmin245,
            '{:.1f}'.format(ntmin245),
            rotation=90,
            va='top',
            ha='center',
            fontsize=12)

        axes[1].text(
            1, ntmin585,
            '{:.1f}'.format(ntmin585),    
            rotation=90,
            va='top',
            ha='center',
            fontsize=12
            )

        axes[1].text(
            0, ntmax245+0.5,
            '{:.1f}'.format(ntmax245),
            rotation=90,
            va='bottom',
            ha='center',
            fontsize=12)

        axes[1].text(
            1, ntmax585+0.5,
            '{:.1f}'.format(ntmax585),    
            rotation=90,
            va='bottom',
            ha='center',
            fontsize=12
            )

        axes[3].bar(
            0, t245_max['2099-12-31']-t245_min['2099-12-31'],
            width=0.7,
            bottom=t245_min['2099-12-31'], alpha=0.3)
        axes[3].bar(
            1, t585_max['2099-12-31']-t585_min['2099-12-31'],
            width=0.7,
            bottom=t585_min['2099-12-31'], alpha=0.3)

        # ax1.add_patch(Rectangle((1.01, 0.2), 0.01, 0.95))

        axes[0].tick_params(axis='both', labelsize=12)
        axes[0].legend(loc='upper left', fontsize=12,)
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['top'].set_visible(False)

        axes[2].spines['left'].set_visible(False)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        axes[2].tick_params(top=False, left=False, right=False)

        # axes[1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        axes[3].text(
            0, (t245_min['2099-12-31']+t245_max['2099-12-31'])/2,
            'ssp245', rotation=90,
            va='top',
            ha='center',
            fontsize=12)
        axes[3].text(
            1, (t585_min['2099-12-31']+t585_max['2099-12-31'])/2,
            'ssp585', rotation=90,
            va='top',
            ha='center',
            fontsize=12
            )

        axes[3].text(
            0, t245_min['2099-12-31'],
            '{:.1f}'.format(t245_min['2099-12-31']),
            rotation=90,
            va='top',
            ha='center',
            fontsize=12)
        axes[3].text(
            0, t245_max['2099-12-31']+0.5,
            '{:.1f}'.format(t245_max['2099-12-31']),
            rotation=90,
            va='bottom',
            ha='center',
            fontsize=12)

        axes[3].text(
            1, t585_min['2099-12-31'],
            '{:.1f}'.format(t585_min['2099-12-31']),
            rotation=90,
            va='top',
            ha='center',
            fontsize=12)

        axes[3].text(
            1, t585_max['2099-12-31']+0.5,
            '{:.1f}'.format(t585_max['2099-12-31']),
            rotation=90,
            va='bottom',
            ha='center',
            fontsize=12)

        ax1.set_ylabel('Annual Average Temperature ($^\circ$C)', fontsize=12)
        # axes[1].legend()
        # ax.set_yscale('log')
        f.tight_layout()
        plt.savefig(os.path.join(self.working_dir, 'dw_scn_tmp.png'), dpi=300, bbox_inches="tight")
        plt.show()
        # ax.plot(base_df.index, base_df, label='BASE')
        # '''
    def plot_mon_rainfall_linear(self):
        df = self.get_pcps()
        # for chirps vs fgoals
        df = df.sort_index().loc['1/1/1983':'12/31/2014', ['chirps_pcp', 'ssp245_pcp', 'ssp585_pcp']]
        msdf = df.resample('ME').sum()

        fig, axes = plt.subplots(2, 5, figsize=(12, 5), sharex=True, sharey=True)
        ax1 = fig.add_subplot(111, frameon=False)
        ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        for ax, c in zip(axes.flat, msdf.columns[1:]):
            ax.plot([0, 400], [0, 400], '--', color='grey', alpha=0.5)
            ax.scatter(msdf.chirps_pcp, msdf[c], alpha=0.3)
            # x_val = msdf.base.tolist()
            # y_val = msdf[c].tolist()
            x_val = msdf[c].tolist()
            y_val = msdf.chirps_pcp.tolist()
            correlation_matrix = np.corrcoef(x_val, y_val)
            correlation_xy = correlation_matrix[0,1]
            r_squared = correlation_xy**2

            m, b = np.polyfit(x_val, y_val, 1)
            ax.plot(np.array(x_val), (m*np.array(x_val)) + b, 'r')

            # ax.set_ylabel('{}'.format('BASE'))
            # ax.set_xlabel('Scenario: {}'.format(c))
            ax.set_title('{}'.format(c))

            ax.text(
                    0.95, 0.05,
                    'R: {:.3f}'.format(r_squared),
                    horizontalalignment='right',
                    bbox=dict(facecolor='red', alpha=0.5),
                    transform=ax.transAxes
                    )
        ax1.set_ylabel('{}'.format('BASE - Monthly Rainfall Intensity $(mm/month)$'), labelpad=10)
        fig.tight_layout()
        plt.savefig(os.path.join(self.working_dir, 'gcms_m_scatter_new.png'), dpi=300, bbox_inches="tight")
        plt.show()

    def create_mon_scn(self):
        #NOTE create scn files monthly base
        working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default"
        # m1 = handler.SWATp(working_dir)
        sn = handler.CliScenario(working_dir)
        scns = []
        for sc in ["245", "585"]:
            for cset in ["hist", "near", "mid", "far"]:
                scns.append(f"ssp{sc}_{cset}")
        fields = ["wateryld", "perc", "et", "sw_ave"]

        # sn.read_lu_wb_csv(scns[0], fields[0])
        for f in fields:    
            sn.get_lu_wb_mon_scns(scns, f)
    
    def create_aa_scn(self):
        working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default"
        scns = []
        for sc in ["245", "585"]:
            for cset in ["hist", "near", "mid", "far"]:
                scns.append(f"ssp{sc}_{cset}")
        for scn in scns:
            m1 = handler.SWATp(os.path.join(working_dir, scn))
            m1.get_lu_wb_aa()

    def scenario_fdc_org(self, df):
        # df from get_streamdischarge_scns
        # from get_stf_sim_obd
        fig, ax = plt.subplots()
        for col in df.columns:

        # odd, oeexd = handler.convert_fdc_data(df.iloc[:, 1].values)
            sdd, seexd = analyzer.convert_fdc_data(df.loc[:, col].dropna().values)

            ax.plot(seexd*100, sdd, lw=2, label=col)
            # ax.scatter(seexd*100, sdd, label=col)
            # ax.plot(oeexd*100, odd, lw=2, label="obd")
            ax.set_yscale('log')
        # ax.set_xlabel(r"Exceedence [%]", fontsize=12)
        # ax.set_ylabel(r"Flow rate $[m^3/s]$", fontsize=12)
        ax.margins(0.01)
        # ax.tick_params(axis='both', labelsize=12)
        plt.legend(fontsize=12, loc="lower left")
        # ax.text(
        #     1, 0.8, f'rel{rel_idx}', fontsize=10,
        #     horizontalalignment='right',
            # transform=ax.transAxes)
        plt.tight_layout()
        # plt.savefig(f'fdc_{obgnme}.png', bbox_inches='tight', dpi=300)
        plt.show()    

    def scenario_fdc(self, df):
        """plot fdc for stream discharge

        :param df: data read from read_stf_scns function
        :type df: dataframe
        """

        from scipy.ndimage import gaussian_filter1d
        # df from get_streamdischarge_scns
        # from get_stf_sim_obd
        fig, ax = plt.subplots(figsize=(7,6))
        for col in df.columns:
            sdd, seexd = analyzer.convert_fdc_data(df.loc[:, col].dropna().values)
            ysmoothed = gaussian_filter1d(sdd, sigma=2)
            if col == "base":
                col = "Historical"
            ax.plot(seexd*100, ysmoothed, lw=2, label=col)
            ax.set_yscale('log')
            # ax.set_xscale('log')
        ax.set_xlabel(r"Exceedence [%]", fontsize=12)
        ax.set_ylabel(r"Flow rate $[m^3/s]$", fontsize=12)
        ax.margins(0.01)
        ax.tick_params(axis='both', labelsize=12)
        plt.legend(fontsize=12, loc="lower left")
        obgnme = "dawhenya"
        plt.tight_layout()
        plt.savefig(os.path.join(self.working_dir, f'fdc_{obgnme}.png'), bbox_inches='tight', dpi=300)
        plt.show()    

    def scenario_fdc_details(self, df):
        from scipy.ndimage import gaussian_filter1d
        # df from get_streamdischarge_scns
        # from get_stf_sim_obd
        fig, axes = plt.subplots(
            1, 3, figsize=(5, 5), 
            gridspec_kw={'width_ratios': [1, 3, 1], 
                        # 'wspace': 0.2
                        }
            )
        for ax in axes:
            for col in df.columns:
                sdd, seexd = analyzer.convert_fdc_data(df.loc[:, col].dropna().values)
                ysmoothed = gaussian_filter1d(sdd, sigma=2)
                ax.plot(seexd*100, ysmoothed, lw=2, label=col)
        # ax.set_xlabel(r"Exceedence [%]", fontsize=12)
        # ax.set_ylabel(r"Flow rate $[m^3/month]$", fontsize=12)
        axes[0].set_xlim(-1, 10)
        # axes[0].set_ylim(250, 1500) #Mun
        axes[0].set_ylim(5, 40)

        axes[1].set_xlim(45, 55)
        # axes[1].set_ylim(100, 160) # mun
        axes[1].set_ylim(1.5, 3.5)
        axes[2].set_xlim(90, 101)
        # axes[2].set_ylim(15, 80)# mun
        axes[2].set_ylim(0.04, 0.5)
        axes[2].yaxis.tick_right()
        for ax in axes:
            ax.tick_params(axis='both', labelsize=12)
            ax.set_yscale('log')
            ax.margins(0.01)
        axes[1].tick_params(axis='y', which='minor', rotation=90)
        # ax.spines[['left', 'top']].set_visible(False)
        # ax.xaxis.tick_bottom()
        obgnme = "dawhenya"
        # fig.subplots_adjust(wspace=0.1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.working_dir, f'fdc_small_{obgnme}.png'), bbox_inches='tight', dpi=300)
        plt.show()


    def plot_violin_fdc_details(self, working_dir, chaid):
        cs = handler.CliScenario(working_dir)
        df = cs.read_stf_scns(chaid)
        # 10 percent
        dff = pd.DataFrame()
        for col in df.columns:
            data, exd = analyzer.convert_fdc_data(df.loc[:, col].dropna().values)
            idx_ext = np.argmax(exd * 100 > 10)
            value_ext = data[idx_ext-1]
            ext_df = df.loc[df[col]>value_ext, col]
            dff = pd.concat([dff, ext_df], axis=1)
        dff.drop("ssp585_hist", axis=1, inplace=True)
        dff.rename({'ssp245_hist': 'base'}, axis=1, inplace=True)
        data = [dff[i].dropna() for i in dff.columns]



        # 90 percent
        dff90 = pd.DataFrame()
        for col in df.columns:
            data90, exd90 = analyzer.convert_fdc_data(df.loc[:, col].dropna().values)
            idx_ext90 = np.argmax(exd90 * 100 > 90)
            value_ext90 = data90[idx_ext90-1]
            ext_df90 = df.loc[df[col]<value_ext90, col]
            dff90 = pd.concat([dff90, ext_df90], axis=1)
        dff90.drop("ssp585_hist", axis=1, inplace=True)
        dff90.rename({'ssp245_hist': 'base'}, axis=1, inplace=True)
        data90 = [dff90[i].dropna() for i in dff90.columns]

        xlabels = ["Historical" if xc == "base" else xc for xc in dff.columns]

        f, axes = plt.subplots(2, 1, figsize=(5,7), sharex=True)
        # ax.boxplot(data, flierprops=flierprops)
        r = axes[0].violinplot(
            data,  widths=0.7, showmeans=True, showextrema=True,
            # bw_method='silverman'
            )
        r90 = axes[1].violinplot(
            data90,  widths=0.7, showmeans=True, showextrema=True,
            # bw_method='silverman'
            )
        r['cmeans'].set_color('r')
        r90['cmeans'].set_color('r')


        # Set the color of the violin patches
        colors = [f"C{i}" for i in range(8)]
        for pc, color in zip(r['bodies'], colors):
            pc.set_facecolor(color)
        for pc, color in zip(r90['bodies'], colors):
            pc.set_facecolor(color)
        for ax in axes:
            ax.set_xticks([i+1 for i in range(len(xlabels))])
            ax.set_xticklabels(xlabels, rotation=90)
            ax.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, 'fdc_violin_details.png'), dpi=300, bbox_inches="tight")
        plt.show()

    def extreme_events(self, df, ext_th): # number of extreme events annual
        for col in df.columns:
            data, exd = analyzer.convert_fdc_data(df.loc[:, col].dropna().values)
            idx_ext = np.argmax(exd * 100 > ext_th)
            value_ext = data[idx_ext-1]
        dft = pd.DataFrame()
        # extreme
        ext_df = df[df >value_ext].groupby(df.index.year).agg('count')

        return ext_df

    def spi(self, df, colnam, interval=90):
        series = df.loc[:, colnam].rolling(interval, min_periods=interval).sum().dropna()
        return series


    def extreme_events2(self, df, ext_val):
        ext_df = df[df >ext_val].groupby(df.index.year).agg('count').mean()
        return ext_df
        # print(ext_df)







    # def plot_bar_landuse_wb(self):

def get_extreme_events():
    # working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\climate_scenarios\\dawhenya\\FGOALS-g3_SWAT_files"
    working_dir = "D:\\Projects\\Watersheds\\Mun\\climate_scenarios\\FGOALS-g3_SWAT_files"
    ca = ClimateAnalyzer(working_dir)
    df = ca.get_pcps()
    timesets = [
        ("1/1/1985", "12/31/2014"),
        ("1/1/2015", "12/31/2040"),
        ("1/1/2041", "12/31/2070"),
        ("1/1/2071", "12/31/2100"),
    ]
    # set 5% threshold from fdc in historical data
    hist_df = df[f"{timesets[0][0]}":f"{timesets[0][1]}"]    
    data, exd = analyzer.convert_fdc_data(hist_df.iloc[:, 0].dropna().values)
    idx_ext = np.argmax(exd * 100 > 5)
    value_ext = data[idx_ext-1]
    print(idx_ext)
    print(value_ext)

    for ts in timesets:
        edf = ca.extreme_events2(df[f"{ts[0]}":f"{ts[1]}"], value_ext)
        print(edf)



def test():
    #NOTE create scn files monthly base
    working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default"
    # m1 = handler.SWATp(working_dir)
    sn = handler.CliScenario(working_dir)
    scns = []
    for sc in ["245", "585"]:
        for cset in ["hist", "near", "mid", "far"]:
            scns.append(f"ssp{sc}_{cset}")

    # sn.read_lu_wb_csv(scns[0], fields[0])

    sn.get_lu_wb_aa_scns(scns)


def fdc_figures():
    # working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default"
    working_dir = "D:\\Projects\\Watersheds\\Mun\\Mun_river_082024\\Scenarios\\Default"
    # scns = []
    # for sc in ["245", "585"]:
    #     for cset in ["hist", "near", "mid", "far"]:
    #         scns.append(f"ssp{sc}_{cset}")
    scns = []
    for sc in ["245", "585"]:
        for cset in ["hist", "near", "mid", "far"]:
            scns.append(f"ssp{sc}_{cset}")
    
    sn = handler.CliScenario(working_dir)
    # sn.extract_stf_day_scns(working_dir, scns, 1, timestep='day')

    # read scns file
    df = sn.read_stf_scns(1)
    df.drop("ssp585_hist", axis=1, inplace=True)
    df.rename({'ssp245_hist': 'base'}, axis=1, inplace=True)
    ClimateAnalyzer(working_dir).scenario_fdc(df)
    ClimateAnalyzer(working_dir).plot_violin_fdc_details(working_dir, 1)


def spi_figure():
    # working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\climate_scenarios\\dawhenya\\FGOALS-g3_SWAT_files"
    working_dir = "D:\\Projects\\Watersheds\\Mun\\climate_scenarios\\FGOALS-g3_SWAT_files"
    sc = ClimateAnalyzer(working_dir)
    df = sc.get_pcps()
    series = sc.spi(df, "ssp585_pcp")
    # spi3_gamma = si.spi(series, dist=scs.gamma, fit_freq="ME")
    spi3_pearson = si.spi(series, dist=scs.pearson3, fit_freq="ME")
    f, ax = plt.subplots(4, 1, figsize=(12, 8))
    # choose a colormap to your liking:
    si.plot.si(spi3_pearson, ax=ax[0], cmap="vik_r")
    si.plot.si(spi3_pearson, ax=ax[1], cmap="vik_r")
    si.plot.si(spi3_pearson, ax=ax[2], cmap="vik_r")
    si.plot.si(spi3_pearson, ax=ax[3], cmap="vik_r")
    ax[0].set_xlim(pd.to_datetime(["1985", "2014"]))
    ax[1].set_xlim(pd.to_datetime(["2015", "2040"]))
    ax[2].set_xlim(pd.to_datetime(["2041", "2070"]))
    ax[3].set_xlim(pd.to_datetime(["2071", "2100"]))
    [x.grid() for x in ax]
    [ax[i].set_ylabel(n, fontsize=14) for i, n in enumerate(["Historical", "Near Future", "Mid-Future", "Far Future"])];
    for axx in ax:
        axx.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, 'spi_585.png'), dpi=300, bbox_inches="tight")
    plt.show()


def spi_probability():
    working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\climate_scenarios\\dawhenya\\FGOALS-g3_SWAT_files"
    # working_dir = "D:\\Projects\\Watersheds\\Mun\\climate_scenarios\\FGOALS-g3_SWAT_files"
    sc = ClimateAnalyzer(working_dir)
    df = sc.get_pcps()
    series = sc.spi(df, "ssp245_pcp")
    spi3_pearson = si.spi(series, dist=scs.pearson3, fit_freq="ME")
    timesets = [
        ("1/1/1985", "12/31/2014"),
        ("1/1/2015", "12/31/2040"),
        ("1/1/2041", "12/31/2071"),
        ("1/1/2071", "12/31/2100"),
    ]
    for ts in timesets:
        print(ts)
        spi3_p_clip = spi3_pearson[f"{ts[0]}":f"{ts[1]}"]
        # Define the ranges
        ranges = [
            (-10, -2), (-2, -1.5), (-1.5, -1.0), (-1.0, 1.0),
            (1.0, 1.5), (1.5, 2.0), (2.0, 10)]

        # Calculate probabilities for each range
        for r in reversed(ranges):
            prob = len(spi3_p_clip[(spi3_p_clip >= r[0]) & (spi3_p_clip < r[1])]) / len(spi3_p_clip)
            print(f"Probability in range {r}: {prob*100:.1f}")

# Example usage
if __name__ == "__main__":
    # working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\climate_scenarios\\dawhenya\\FGOALS-g3_SWAT_files"
    working_dir = "D:\\Projects\\Watersheds\\Mun\\climate_scenarios\\FGOALS-g3_SWAT_files"
    # # sn = handler.CliScenario(working_dir)
    # # # sn.extract_stf_day_scns(working_dir, scns, 1, timestep='day')

    # # # read scns file
    # # df = sn.read_stf_scns(1)
    # # df.drop("ssp585_hist", axis=1, inplace=True)
    # # df.rename({'ssp245_hist': 'base'}, axis=1, inplace=True)
    # # ClimateAnalyzer(working_dir).scenario_fdc(df)

    sc = ClimateAnalyzer(working_dir)
    sc.plot_pcps()
    # sc.plot_temps()

    # get_extreme_events()
    # spi_figure()





