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
from swatp_pst import handler


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
        base.name = "base"
        nf = df['1/1/2015':'12/31/2040'].loc[:, ["ssp245_pcp", "ssp585_pcp"]]
        nf.columns = ["ssp245_nf", "ssp585_nf"]
        mf = df['1/1/2041':'12/31/2070'].loc[:, ["ssp245_pcp", "ssp585_pcp"]]
        mf.columns = ["ssp245_mf", "ssp585_mf"]
        ff = df['1/1/2071':'12/31/2099'].loc[:, ["ssp245_pcp", "ssp585_pcp"]]
        ff.columns = ["ssp245_ff", "ssp585_ff"]
        dff = pd.concat([base, nf, mf, ff], axis=1)
        xlabels = [x for x in dff.columns]


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
        

        print(msdf)

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


# Example usage
if __name__ == "__main__":
    
    # working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\climate_scenarios\\dawhenya\\FGOALS-g3_SWAT_files"
    # m1 = ClimateAnalyzer(working_dir)
    # m1.plot_pcps()
    # cli_down()

    # fields = ["wateryld", "perc", "et", "sw_ave", "latq_runon"]
    # # for f in fields:
    # #     m1.get_lu_mon(f)

    working_dir = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default"
    # scns = []
    # for sc in ["245", "585"]:
    #     for cset in ["hist", "near", "mid", "far"]:
    #         scns.append(f"ssp{sc}_{cset}")
    scns = []
    for sc in ["245", "585"]:
        for cset in ["hist", "near", "mid", "far"]:
            scns.append(f"ssp{sc}_{cset}")
    
    sn = handler.CliScenario(working_dir)
    sn.get_lu_wb_aa_scns(scns)
    
    # for scn in scns:
    #     m1 = handler.SWATp(os.path.join(working_dir, scn))
    #     m1.read_lu_wb_aa_csv()

    



