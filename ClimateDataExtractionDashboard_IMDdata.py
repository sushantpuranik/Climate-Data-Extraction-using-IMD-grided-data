import io
import re
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Climate Data Extraction Dashboard",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ Climate Data Time Series Extractor (Tmax, Tmin, Rainfall)")
st.markdown(
    """
Extract multi-year daily time series for specific geographic coordinates. 
Choose your parameters, upload the yearwise binary/gridded data files, and export the extracted data as a CSV.
"""
)

# -----------------------------------------------------------------------------
# SIDEBAR: Grid & File Settings
# -----------------------------------------------------------------------------
st.sidebar.header("1. Grid Configuration")

lat_min = st.sidebar.number_input(
    "Min Latitude (°N)", value=6.5, step=0.25, format="%.2f"
)
lat_max = st.sidebar.number_input(
    "Max Latitude (°N)", value=38.5, step=0.25, format="%.2f"
)
lon_min = st.sidebar.number_input(
    "Min Longitude (°E)", value=66.5, step=0.25, format="%.2f"
)
lon_max = st.sidebar.number_input(
    "Max Longitude (°E)", value=100.0, step=0.25, format="%.2f"
)
resolution = st.sidebar.number_input(
    "Grid Resolution (°)", value=0.25, step=0.05, format="%.2f"
)
no_data_value = st.sidebar.number_input("No-Data Value", value=99.9, step=0.1)

# Generate coordinate arrays for the grid
lats = np.arange(lat_min, lat_max + resolution / 2, resolution)
lons = np.arange(lon_min, lon_max + resolution / 2, resolution)
n_lat = len(lats)
n_lon = len(lons)

st.sidebar.info(f"Grid Size: **{n_lat} Lats × {n_lon} Lons**")

# -----------------------------------------------------------------------------
# PARAMETER SELECTION & FILE UPLOADERS
# -----------------------------------------------------------------------------
st.header("2. Select Parameters & Upload Files")

selected_params = st.multiselect(
    "Choose parameters to extract:",
    ["Tmax", "Tmin", "Rainfall"],
    default=["Tmax", "Tmin"],
)

col1, col2, col3 = st.columns(3)
tmax_files, tmin_files, rf_files = [], [], []

if "Tmax" in selected_params:
  with col1:
    tmax_files = st.file_uploader(
        "Upload Tmax Files (e.g., Maxtemp_MaxT_2022)",
        accept_multiple_files=True,
        key="tmax",
    )

if "Tmin" in selected_params:
  with col2:
    tmin_files = st.file_uploader(
        "Upload Tmin Files (e.g., Mintemp_MinT_2022)",
        accept_multiple_files=True,
        key="tmin",
    )

if "Rainfall" in selected_params:
  with col3:
    rf_files = st.file_uploader(
        "Upload Rainfall Files (e.g., Rainfall_ind1971_rfp25)",
        accept_multiple_files=True,
        key="rf",
    )


# Function to parse year from filename (supports 1971, 2022, etc.)
# Updated function to reliably extract 4-digit years (19xx or 20xx)
def extract_year(filename):
  match = re.search(r'(19\d\d|20\d\d)', filename)
  return int(match.group(1)) if match else None

# Function to read binary grid year data securely
def read_binary_year_data(uploaded_file, n_lat, n_lon, no_data_val):
  year = extract_year(uploaded_file.name)
  if year is None:
    return None, None, None

  is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
  n_days = 366 if is_leap else 365

  # Reset Streamlit file pointer
  uploaded_file.seek(0)
  bytes_data = uploaded_file.read()
  data = np.frombuffer(bytes_data, dtype=np.float32)

  expected_size = n_days * n_lat * n_lon
  if data.size != expected_size:
    st.error(
        f"File {uploaded_file.name} size mismatch. Expected {expected_size}"
        f" floats, got {data.size}."
    )
    return None, None, None

  # Reshape and make writable using .copy()
  grid_data = data.reshape((n_days, n_lat, n_lon)).copy()
  grid_data[np.isclose(grid_data, no_data_val)] = np.nan

  dates = pd.date_range(start=f"{year}-01-01", periods=n_days, freq="D")
  return year, dates, grid_data


# -----------------------------------------------------------------------------
# COORDINATE SELECTION
# -----------------------------------------------------------------------------
st.header("3. Target Coordinates Input")
input_mode = st.radio(
    "Select Input Method:",
    ["Single Coordinate (Manual)", "Batch Coordinates (CSV Upload)"],
)

targets = []

if input_mode == "Single Coordinate (Manual)":
  c1, c2, c3 = st.columns(3)
  point_name = c1.text_input("Point Name/ID", value="Location_1")
  target_lat = c2.number_input(
      "Latitude (°N)",
      value=float(lats[len(lats) // 2]),
      min_value=float(lat_min),
      max_value=float(lat_max),
  )
  target_lon = c3.number_input(
      "Longitude (°E)",
      value=float(lons[len(lons) // 2]),
      min_value=float(lon_min),
      max_value=float(lon_max),
  )
  targets.append(
      {"ID": point_name, "Latitude": target_lat, "Longitude": target_lon}
  )

else:
  csv_file = st.file_uploader(
      "Upload CSV File (Columns must include: `ID`, `Latitude`, `Longitude`)",
      type=["csv"],
  )
  if csv_file:
    df_coord = pd.read_csv(csv_file)
    st.dataframe(df_coord.head())
    if (
        "Latitude" in df_coord.columns
        and "Longitude" in df_coord.columns
        and "ID" in df_coord.columns
    ):
      targets = df_coord.to_dict("records")
    else:
      st.error("CSV must contain `ID`, `Latitude`, and `Longitude` columns.")

# -----------------------------------------------------------------------------
# EXTRACTION & OUTPUT GENERATION
# -----------------------------------------------------------------------------
if st.button("🚀 Extract Time Series"):
  if not selected_params:
    st.warning("Please select at least one parameter to extract.")
  elif (
      ("Tmax" in selected_params and not tmax_files)
      or ("Tmin" in selected_params and not tmin_files)
      or ("Rainfall" in selected_params and not rf_files)
  ):
    st.warning("Please upload files for all selected parameters.")
  elif not targets:
    st.warning("Please specify target coordinate(s).")
  else:
    with st.spinner("Processing binary files and extracting time series..."):
      # Map uploaded files by year
      # Map uploaded files by year (Passing f.name instead of f)
      tmax_dict = (
          {extract_year(f.name): f for f in tmax_files if extract_year(f.name)}
          if "Tmax" in selected_params
          else {}
      )
      tmin_dict = (
          {extract_year(f.name): f for f in tmin_files if extract_year(f.name)}
          if "Tmin" in selected_params
          else {}
      )
      rf_dict = (
          {extract_year(f.name): f for f in rf_files if extract_year(f.name)}
          if "Rainfall" in selected_params
          else {}
      )

      # Determine overlapping/common years among selected parameters
      available_sets = []
      if "Tmax" in selected_params:
        available_sets.append(set(tmax_dict.keys()))
      if "Tmin" in selected_params:
        available_sets.append(set(tmin_dict.keys()))
      if "Rainfall" in selected_params:
        available_sets.append(set(rf_dict.keys()))

      if available_sets:
        common_years = sorted(list(set.intersection(*available_sets)))
      else:
        common_years = []

      if not common_years:
        st.error(
            "No matching years found across the selected parameter files."
            " Check your filenames to ensure standard 4-digit years (e.g.,"
            " 1971, 2022) are present."
        )
      else:
        all_dfs = []

        for year in common_years:
          grid_max, grid_min, grid_rf = None, None, None
          dates = None

          if "Tmax" in selected_params:
            _, dates, grid_max = read_binary_year_data(
                tmax_dict[year], n_lat, n_lon, no_data_value
            )
          if "Tmin" in selected_params:
            _, dates, grid_min = read_binary_year_data(
                tmin_dict[year], n_lat, n_lon, no_data_value
            )
          if "Rainfall" in selected_params:
            _, dates, grid_rf = read_binary_year_data(
                rf_dict[year], n_lat, n_lon, no_data_value
            )

          if dates is None:
            continue

          # Extract time series for each target coordinate
          for target in targets:
            tid = target["ID"]
            plat = target["Latitude"]
            plon = target["Longitude"]

            lat_idx = (np.abs(lats - plat)).argmin()
            lon_idx = (np.abs(lons - plon)).argmin()

            row_data = {
                "Date": dates,
                "Location_ID": tid,
                "Target_Lat": plat,
                "Target_Lon": plon,
                "Nearest_Lat": lats[lat_idx],
                "Nearest_Lon": lons[lon_idx],
            }

            if "Tmax" in selected_params and grid_max is not None:
              row_data["Tmax"] = grid_max[:, lat_idx, lon_idx]
            if "Tmin" in selected_params and grid_min is not None:
              row_data["Tmin"] = grid_min[:, lat_idx, lon_idx]
            if "Rainfall" in selected_params and grid_rf is not None:
              row_data["Rainfall"] = grid_rf[:, lat_idx, lon_idx]

            df_year = pd.DataFrame(row_data)
            all_dfs.append(df_year)

        if all_dfs:
          final_df = pd.concat(all_dfs, ignore_index=True)

          st.success("Extraction Complete!")

          st.subheader("Data Preview")
          st.dataframe(final_df.head(20))

          csv_buffer = io.BytesIO()
          final_df.to_csv(csv_buffer, index=False)
          csv_bytes = csv_buffer.getvalue()

          st.download_button(
              label="📥 Download Daily Time Series CSV",
              data=csv_bytes,
              file_name="extracted_climate_time_series.csv",
              mime="text/csv",
          )