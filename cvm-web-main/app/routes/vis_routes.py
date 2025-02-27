import os
import sys
import json
import traceback

from fastapi import APIRouter, Request, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from utils.json_utilities import read_json_files
import xarray as xr
from pyproj import Proj, Geod
from numpy import nanmin, nanmax

# from metpy.interpolate import cross_section
import base64
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from datetime import datetime, timezone
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

import math
import logging
import gc

import urllib.parse
from uuid import uuid4


from geographiclib.geodesic import Geodesic


router = APIRouter()

# Get the absolute path of the directory where the script is located
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# Get the parent directory
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

# Set the base directory to the parent directory
BASE_DIR = PARENT_DIR
JSON_DIR = os.path.join(BASE_DIR, "static", "json")
NETCDF_DIR = os.path.join(BASE_DIR, "static", "netcdf")
LOGO_FILE = "static/images/Crescent_Logos_horizontal_transparent.png"

# Logo placement in inches below the lower left corner of the plot area
LOGO_INCH_OFFSET_BELOW = 0.5

grid_mapping_dict = {
    "latitude_longitude": {
        "x": "longitude",
        "y": "latitude",
        "x2": "longitude",
        "y2": "latitude",
    },
    "transverse_mercator": {"x": "x", "y": "y", "x2": "easting", "y2": "northing"},
}

# Templates
templates = Jinja2Templates(directory="templates")


def points_to_array(points, delimiter=";"):
    """Converts a string of coordinate points that are separated by a delimiter to array of points.'123,45;124,46' -> [[123,45],[124,46]]"""
    points_list = points.strip().split(delimiter)
    coord_list = list()
    name_list = list()
    name_info_list = list()
    for point in points_list:
        lon_lat, name, name_info = point.strip().split("|")
        lon, lat = lon_lat.strip().split(",")
        coord_list.append([float(lat), float(lon)])
        name_list.append(name)
        name_info_list.append(name_info)
    return coord_list, name_list, name_info_list


def points_to_dist(start, points_list):
    """calculates distance of a series of points from a starting point."""
    distance_list = list()
    geod = Geod(ellps="WGS84")

    for point in points_list:
        _, _, distance = geod.inv(start[1], start[0], point[1], point[0])
        distance_list.append(distance)
    return distance_list


def utc_now():
    """Return the current UTC time."""
    try:
        _utc = datetime.now(tz=timezone.utc)
        utc = {
            "date_time": _utc.strftime("%Y-%m-%dT%H:%M:%S"),
            "datetime": _utc,
            "epoch": _utc.timestamp(),
        }
        return utc
    except:
        logging.error(f"[ERR] Failed to get the current UTC time")
        raise


def standard_units(unit):
    """Check an input unit and return the corresponding standard unit."""
    unit = unit.strip().lower()
    if unit in ["m", "meter"]:
        return "m"
    elif unit in ["degrees", "degrees_east", "degrees_north"]:
        return "degrees"
    elif unit in ["km", "kilometer"]:
        return "km"
    elif unit in ["g/cc", "g/cm3", "g.cm-3", "grams.centimeter-3"]:
        return "g/cc"
    elif unit in ["kg/m3", "kh.m-3"]:
        return "kg/m3"
    elif unit in ["km/s", "kilometer/second", "km.s-1", "kilometer/s", "km/s"]:
        return "km/s"
    elif unit in ["m/s", "meter/second", "m.s-1", "meter/s", "m/s"]:
        return "m/s"
    elif unit.strip().lower in ["", "none"]:
        return ""


def unit_conversion_factor(unit_in, unit_out):
    """Check input and output unit and return the conversion factor."""

    unit = standard_units(unit_in.strip().lower())
    logging.warning(f"[INFO] convert units {unit} to {unit_out}")
    if unit in ["m"]:
        if unit_out == "cgs":
            return standard_units("m"), 1
        else:
            return standard_units("km"), 0.001
    elif unit in ["km"]:
        if unit_out == "cgs":
            return standard_units("m"), 1000
        else:
            return standard_units("km"), 1
    elif unit in ["m/s"]:
        if unit_out == "cgs":
            return standard_units("m/s"), 1
        else:
            return standard_units("km/s"), 0.001
    elif unit in ["g/cc"]:
        if unit_out == "cgs":
            return standard_units("g/cc"), 1
        else:
            return standard_units("kg/m3"), 1000
    elif unit in ["km/s"]:
        if unit_out == "cgs":
            return standard_units("m/s"), 1000
        else:
            return standard_units("km/s"), 1
    elif unit in ["kg/m3"]:
        if unit_out == "cgs":
            return standard_units("g/cc"), 0.001
        else:
            return standard_units("kg/m3"), 1
    elif unit in ["", " "]:
        return standard_units(""), 1
    elif unit in ["degrees"]:
        return standard_units("degrees"), 1

    else:
        logging.error(f"[ERR] Failed to convert units {unit_in} to {unit_out}")


def dip_dir_to_azimuth(dip_dir):
    """Convert dip direction to azimuth angle."""
    cardinal_to_azimuth = {
        "N": 0,
        "NE": 45,
        "E": 90,
        "SE": 135,
        "S": 180,
        "SW": 225,
        "W": 270,
        "NW": 315,
    }
    if dip_dir.upper() == "VERTICAL":
        return None  # Special case for vertical dip direction
    return cardinal_to_azimuth[dip_dir.upper()]


def calculate_azimuth(start_lat, start_lon, end_lat, end_lon):
    """Calculate azimuth between two geographical points."""
    geod = Geodesic.WGS84
    inv = geod.Inverse(start_lat, start_lon, end_lat, end_lon)
    return inv["azi1"]


def project_feature(
    dip, dip_azimuth, section_azimuth, upper_depth, lower_depth, start_lat, start_lon
):
    """Project the feature onto the section plane, accounting for dip."""
    dip_angle = np.radians(dip)
    vertical_range = lower_depth - upper_depth

    if dip_azimuth is None:
        # Vertical dip: horizontal positions are the same
        horizontal_position_start = 0
        horizontal_position_end = 0
    else:
        relative_azimuth = np.radians(dip_azimuth - section_azimuth)
        horizontal_offset = vertical_range / np.tan(dip_angle)
        horizontal_position_start = 0  # Start position relative to start coordinates
        horizontal_position_end = horizontal_offset * np.cos(relative_azimuth)

    return horizontal_position_start, horizontal_position_end, upper_depth, lower_depth


def interpolate_path(
    ds,
    start,
    end,
    num_points=100,
    method="linear",
    grid_mapping="latitude_longitude",
    utm_zone=None,
    ellipsoid=None,
):
    """
    Interpolates a dataset along a path defined by start and end coordinates on an irregular grid.

    Parameters:
        ds (xarray.Dataset): The input dataset containing 'latitude' and 'longitude' as coordinates.
        start (tuple): A tuple (latitude, longitude) of the starting point.
        end (tuple): A tuple (latitude, longitude) of the ending point.
        num_points (int): Number of points to interpolate along the path.
        method (str): Interpolation method to use ('linear', 'nearest').

    Returns:
        xarray.Dataset: The interpolated dataset along the path.
    """
    # Create linearly spaced points between start and end
    lat_points = np.linspace(start[0], end[0], num_points)
    lon_points = np.linspace(start[1], end[1], num_points)

    # Define a path dataset for interpolation
    path = xr.Dataset(
        {"latitude": ("points", lat_points), "longitude": ("points", lon_points)}
    )

    # Interpolate the dataset to these points using the specified method
    if grid_mapping == "latitude_longitude":
        interpolated_ds = ds.interp(
            latitude=path.latitude, longitude=path.longitude, method=method
        )
    else:
        if None in (utm_zone, ellipsoid):
            message = f"[ERR] for grid_mapping: {grid_mapping}, utm_zone and ellipsoid are required. Current values: {utm_zone}, {ellipsoid}!"
            logging.error(message)
            raise
        x_points = list()
        y_points = list()
        for index, lat_value in enumerate(lat_points):
            x, y = project_lonlat_utm(
                lon_points[index], lat_points[index], utm_zone, ellipsoid=ellipsoid
            )
            x_points.append(x)
            y_points.append(y)

        # Define a path dataset for interpolation
        path = xr.Dataset({"x": ("points", x_points), "y": ("points", y_points)})
        logging.warn(f"[INFO] x_points: {x_points}")
        logging.warn(f"[INFO] y_points: {y_points}")
        interpolated_ds = ds.interp(x=path.x, y=path.y, method=method)

    return interpolated_ds, lat_points, lon_points


def project_lonlat_utm(
    longitude, latitude, utm_zone, ellipsoid, xy_to_latlon=False, preserve_units=False
):
    """
    Performs cartographic transformations. Converts from longitude, latitude to UTM x,y coordinates
    and vice versa using PROJ (https://proj.org).

     Keyword arguments:
    longitude (scalar or array) – Input longitude coordinate(s).
    latitude (scalar or array) – Input latitude coordinate(s).
    xy_to_latlon (bool, default=False) – If inverse is True the inverse transformation from x/y to lon/lat is performed.
    preserve_units (bool) – If false, will ensure +units=m.
    """
    P = Proj(
        proj="utm",
        zone=utm_zone,
        ellps=ellipsoid,
    )
    # preserve_units=preserve_units,

    x, y = P(
        longitude,
        latitude,
        inverse=xy_to_latlon,
    )
    return x, y


def closest(lst, value):
    """Find the closest number in a list to the given value

    Keyword arguments:
    lst -- [required] list of the numbers
    value -- [required] value to find the closest list member for.
    """
    arr = np.array(lst)
    # Check if the array is multi-dimensional
    if arr.ndim > 1:
        # Flatten the array and return
        flat_list = list(arr.flatten())
    else:
        # Return the original array if it's already one-dimensional
        flat_list = lst

    return flat_list[
        min(range(len(flat_list)), key=lambda i: abs(flat_list[i] - value))
    ]


def create_error_image(message: str) -> BytesIO:
    """Generate an image with an error message."""
    # Create an image with white background
    img = Image.new("RGB", (600, 100), color=(255, 255, 255))

    # Initialize the drawing context
    d = ImageDraw.Draw(img)

    # Optionally, add a font (this uses the default PIL font)
    # For custom fonts, use ImageFont.truetype()
    # font = ImageFont.truetype("arial.ttf", 15)

    # Add text to the image
    d.text(
        (10, 10), message, fill=(255, 0, 0)
    )  # Change coordinates and color as needed

    # Save the image to a bytes buffer
    img_io = BytesIO()
    img.save(img_io, "PNG")
    img_io.seek(0)
    image_data = (
        img_io.read()
    )  # Read the entire stream content, which is the image data

    return image_data


def custom_formatter(x):
    """
    Custom formatter function
    If the value is close enough to zero (including -0.0), format it as '0'.
    Otherwise, use the default formatting.
    """
    if abs(x) < 1e-12:  # 1e-12 is used as a threshold for floating-point comparison
        return "0"
    else:
        return f"{x}"


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/test3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("test_3d.html", {"request": request})

@router.get("/div", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("div.html", {"request": request})


@router.get("/x-section", response_class=HTMLResponse)
async def read_page1(request: Request):
    return templates.TemplateResponse("x-section.html", {"request": request})


@router.get("/depth-slice-viewer", response_class=HTMLResponse)
async def read_page1(request: Request):
    return templates.TemplateResponse("depth-slice-viewer.html", {"request": request})

@router.get("/depth-slice-3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("depth-slice-3d.html", {"request": request})


# Route to create html for model dropdown.
@router.get("/models_drop_down", response_class=HTMLResponse)
def models_drop_down():
    init_model = "Cascadia-ANT+RF-Delph2018.r0.1.nc"
    json_directory = JSON_DIR
    netcdf_directory = NETCDF_DIR
    file_list = list()
    vars_list = list()
    model_list = list()
    for filename in sorted(os.listdir(json_directory)):
        nc_filename = filename.replace(".json", ".nc")
        filepath = os.path.join(netcdf_directory, nc_filename)

        if os.path.isfile(filepath):
            file_list.append(nc_filename)
            # Opening the file and loading the data
            with open(os.path.join(json_directory, filename), "r") as file:
                json_data = json.load(file)
                data_vars = f"({','.join(json_data['data_vars'])})"
                model_name = json_data["model"]
                vars_list.append(data_vars)
                model_list.append(model_name)

    # Prepare the HTML for the dropdown
    dropdown_html = ""
    for i, filename in enumerate(file_list):
        selected = " selected" if filename == init_model else ""
        dropdown_html += f'<option value="{filename}"{selected}>{model_list[i]} {vars_list[i]}</option>'
    return dropdown_html


# Route to display the table of JSON files with auxiliary information hidden.
@router.get("/list_json_files", response_class=HTMLResponse)
def list_table(request: Request):
    html_content = """
    <table>
    <tr>
    <th></th>
    <th>Model</th>
    <th>Summary</th>
    <th class='hidden'>lat_min</th>
    <th class='hidden'>lat_max</th>
    <th class='hidden'>lon_min</th>
    <th class='hidden'>lon_max</th>
    <th class='hidden'>file</th>
    </tr>
    """

    directory = JSON_DIR
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(".json"):
            with open(filepath, "r") as file:
                data = json.load(file)
                lon_min = data.get("geospatial_lon_min", "-")
                lon_max = data.get("geospatial_lon_max", "-")
                lat_min = data.get("geospatial_lat_min", "-")
                lat_max = data.get("geospatial_lat_max", "-")
                model = data.get("model", "-")
                summary = data.get("summary", "-")
                filename_no_extension = os.path.splitext(filename)[0]
                html_content = f"""{html_content}
                <tr>
                <td style='background-color: white;'><input type='radio' name='selection'></td>
                <td style='text-decoration:underline;cursor:pointer;'>{model}</td>
                <td>{summary}</td>
                <td class='hidden'>{lat_min}</td>
                <td class='hidden'>{lat_max}</td>
                <td class='hidden'>{lon_min}</td>     
                <td class='hidden'>{lon_max}</td>   
                <td class='hidden' name='filename'>{filename_no_extension}</td>
                </tr>
                """

    html_content = f"{html_content}</table>"
    return html_content


def project_lonlat_utm(
    longitude, latitude, utm_zone, ellipsoid, xy_to_latlon=False, preserve_units=False
):
    """
    Performs cartographic transformations. Converts from longitude, latitude to UTM x,y coordinates
    and vice versa using PROJ (https://proj.org).

     Keyword arguments:
    longitude (scalar or array) – Input longitude coordinate(s).
    latitude (scalar or array) – Input latitude coordinate(s).
    xy_to_latlon (bool, default=False) – If inverse is True the inverse transformation from x/y to lon/lat is performed.
    preserve_units (bool) – If false, will ensure +units=m.
    """
    P = Proj(
        proj="utm",
        zone=utm_zone,
        ellps=ellipsoid,
    )
    # preserve_units=preserve_units,

    x, y = P(
        longitude,
        latitude,
        inverse=xy_to_latlon,
    )
    return x, y


def read_image(image_path):
    """Read an image file."""
    try:
        return plt.imread(image_path)
    except Exception as e:
        logging.error(f"[ERR] Failed to read image {image_path}: {e}")
        raise


def get_colormap_names():
    """Returns a list of all colormap names in matplotlib."""
    return sorted(plt.colormaps())


@router.get("/colormaps", response_class=HTMLResponse)
async def colormap_dropdown():
    default = "jet_r"
    colormap_names = get_colormap_names()
    options_html = ""
    for cmap in colormap_names:
        selected = " selected" if cmap == default else ""
        options_html += f'<option value="{cmap}"{selected}>{cmap}</option>'
    html_content = f"""
        <select name="colormap">
            {options_html}
        </select>"""
    return html_content


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


# Plot an interpolated cross-section.
@router.route("/xsection", methods=["POST"])
async def xsection(request: Request):
    version = "v.2024.102"
    version_label = f"{version}  {utc_now()['date_time']} UTC"
    form_data = await request.form()

    items_dict = dict(form_data.items())
    logging.warning(f"[INFO] POST: {items_dict}")

    try:
        # lazy loading with Dask.
        plot_data = xr.open_dataset(
            os.path.join("static", "netcdf", items_dict["data_file"]), chunks={}
        )
        vertical_exaggeration = float(items_dict["vertical_exaggeration"])
        label_faults = False
        if "fault_location" in items_dict:
            if len(items_dict["intersection_coords"].strip()) > 0:
                label_faults = True

        start = [float(items_dict["start_lat"]), float(items_dict["start_lng"])]
        end = [float(items_dict["end_lat"]), float(items_dict["end_lng"])]
        depth = [float(items_dict["start_depth"]), float(items_dict["end_depth"])]
        if label_faults:
            logging.warning(items_dict["intersection_coords"])
            intersection_coords, intersection_names, intersection_name_info = (
                points_to_array(items_dict["intersection_coords"])
            )
            fault_distances = points_to_dist(start, intersection_coords)
            logging.warning(
                f"\n\n[INFO] Fault Distances m {fault_distances}, {intersection_names}, {intersection_name_info}"
            )
        units = items_dict["units"]
        title = items_dict["title"]
        image_width = float(items_dict["image_width"])
        image_height = float(items_dict["image_height"])
        figsize = (image_width, image_height)
        logging.warning(
            f"\n\n[INFO] converting the depth units {plot_data['depth']},\nunits: {plot_data['depth'].attrs['units']}"
        )
        unit_standard, depth_factor = unit_conversion_factor(
            plot_data["depth"].attrs["units"], units
        )
        logging.warning(
            f"[INFO] depth units: {plot_data['depth'].attrs['units']},  {depth_factor}"
        )
        plot_data["depth"].attrs["units"] = unit_standard
        x_label = f"distance ({unit_standard})"
        y_label = f"{plot_data['depth'].attrs['long_name']} ({unit_standard}, VE: {vertical_exaggeration}x )"

        plot_data["depth"] = plot_data["depth"] * depth_factor
        plot_data["depth"].attrs["units"] = units

        plot_data = plot_data.where(
            (plot_data.depth >= float(depth[0])) & (plot_data.depth <= float(depth[1])),
            drop=True,
        )
    except Exception as ex:
        logging.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(
                f"[ERR] Bad selection: \n{ex}\n{traceback.print_exc()}"
            ),
            media_type="image/png",
        )
    utm_zone = None
    meta = plot_data.attrs
    dpi = 100
    fig_width, fig_height = figsize  # Figure size in inches
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    logging.warning(f"\n\n[INFO] META: {meta}")
    if "grid_mapping_name" not in meta:
        logging.warning(
            f"[WARN] The 'grid_mapping_name' attribute not found. Assuming geographic coordinate system"
        )
        grid_mapping_name = "latitude_longitude"
    else:
        grid_mapping_name = meta["grid_mapping_name"]
        """
        if grid_mapping_name == "transverse_mercator":
            if "utm_zone" in meta:
                utm_zone = meta["utm_zone"]
                logging.info(f"[INFO] UTM zone: {utm_zone}")
                projection = ccrs.UTM(utm_zone)
                # We use MetPy’s CF parsing to get the data ready for use, and squeeze down the size-one time dimension.
                plot_data = plot_data.metpy.assign_crs(
                    projection.to_cf()
                )
                plot_data = plot_data.metpy.parse_cf().squeeze()
                plot_data = (
                    plot_data.metpy.assign_latitude_longitude(
                        force=True
                    )
                )
            else:
                message =  f"[ERR] The required attribute 'utm_zone' is missing for the grid_mapping_name of {grid_mapping_name}"
                logging.error(message)
                return Response(content=create_error_image(message), media_type="image/png")

        elif grid_mapping_name == "latitude_longitude":
            projection = ccrs.Geodetic()
            plot_data = plot_data.metpy.assign_crs(
                projection.to_cf()
            )
            plot_data = plot_data.metpy.parse_cf().squeeze()
        else:
            message =   f"[ERR] The grid_mapping_name of {grid_mapping_name} is not supported!"
            logging.error(message)
            return Response(content=create_error_image(message), media_type="image/png")
        """
        # Cross-section interpolation type.
        interp_type = items_dict["interpolation_method"]

        # Steps in the cross-section.
        steps = int(items_dict["num_points"])
        logging.info(
            f"[INFO] cross_section start:{start}, end: {end}, steps: {steps}, interp_type: {interp_type}, plot_data: {plot_data}"
        )
        # Extract the cross-section.
        logging.info(
            f"[INFO] Before cross_section (plot_data,start,end,steps,interp_type:)\n{plot_data},{start},{end},{steps},{interp_type}"
        )
        try:
            plot_data, latitudes, longitudes = interpolate_path(
                plot_data,
                start,
                end,
                num_points=steps,
                method=interp_type,
                grid_mapping=grid_mapping_name,
                utm_zone=meta["utm_zone"],
                ellipsoid=meta["ellipsoid"],
            )
            """
            plot_data = cross_section(
                plot_data,
                start,
                end,
                steps=steps,
                interp_type=interp_type,
            )
            """
        except Exception as ex:
            message = f"[ERR] cross_section failed: {ex}\n{traceback.print_exc()}"
            logging.error(message)
            return Response(content=create_error_image(message), media_type="image/png")

        plot_var = items_dict["plot_variable"]

        logging.warning(
            f"\n\n[INFO] After cross_section: plot_data:{plot_data[plot_var]}"
        )
        logging.warning(
            f"\n\n[INFO] After cross_section: plot_data:{plot_data[plot_var].values}"
        )
        # Extract latitude and longitude from your cross-section data
        # latitudes = plot_data['latitude'].values
        # longitudes = plot_data['longitude'].values

        # If the original is not geographic, going outside the model coverage will result in NaN values.
        # Recompute these using the primary coordinates.
        # for _index, _lon in enumerate(plot_data['latitude'].values):
        #     if  np.isnan(_lon):
        #         plot_data['longitude'].values[_index], plot_data['latitude'].values[_index] = project_lonlat_utm(
        #         plot_data['x'].values[_index], plot_data['y'].values[_index], utm_zone, meta["ellipsoid"], xy_to_latlon=True)

        # Geod object for WGS84 (a commonly used Earth model)
        geod = Geod(ellps="WGS84")

        # Calculate distances between consecutive points
        _, _, distances = geod.inv(
            longitudes[:-1], latitudes[:-1], longitudes[1:], latitudes[1:]
        )

        # Compute cumulative distance, starting from 0
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

        if units == "mks":
            cumulative_distances = cumulative_distances / 1000.0

        logging.warning(
            f"\n\n[INFO] units: {units}, cumulative_distances: {cumulative_distances}"
        )

        # Assuming 'plot_data' is an xarray Dataset or DataArray
        # Create a new coordinate 'distance' based on the cumulative distances
        plot_data = plot_data.assign_coords(distance=("points", cumulative_distances))

        # If you want to use 'distance' as a dimension instead of 'index',
        # you can swap the dimensions (assuming 'index' is your current dimension)
        plot_data = plot_data.swap_dims({"points": "distance"})
        logging.warning(f"\n\n[INFO] plot_data:{plot_data}")
        data_to_plot = plot_data[plot_var]

        # Iterate through the model variables and plot each cross-section.
        cmap = items_dict["colormap"]

        # plot_data["depth"] = -plot_data["depth"]

        vmin = items_dict["start_value"].strip()
        if vmin == "auto":
            vmin = ""

        vmax = items_dict["end_value"].strip()
        if vmax == "auto":
            vmax = ""

        logging.info(
            f"[INFO] plot_var units: {data_to_plot.attrs['units']}, units: {units}"
        )
        unit_standard, var_factor = unit_conversion_factor(
            data_to_plot.attrs["units"], units
        )
        logging.warning(
            f"\n\n[INFO] plot_var units: {data_to_plot.attrs['units']}, units: {units} => unit_standard:{unit_standard}, var_factor: {var_factor}"
        )
        data_to_plot = data_to_plot * var_factor

        data_to_plot.attrs["units"] = unit_standard

        logging.info(f"[INFO] Cross-section input: {items_dict}")
        # logging.info(f"[INFO] Cross-section input: {items_dict['data_file']}, start: {start}, end: {end}, step: {steps}, interp_type: {interp_type}, plot_var: {plot_var}")
        data_to_plot["depth"] = data_to_plot["depth"] * -1
        logging.warning(f"\n\n[INFO] data_to_plot:{data_to_plot}")
        try:
            if vmin and vmax:
                vmin = min(
                    float(items_dict["start_value"]), float(items_dict["end_value"])
                )
                vmax = max(
                    float(items_dict["start_value"]), float(items_dict["end_value"])
                )

                data_to_plot.plot.contourf(
                    x="distance", y="depth", cmap=cmap, vmin=vmin, vmax=vmax
                )
            elif vmin:
                data_to_plot.plot.contourf(
                    x="distance", y="depth", cmap=cmap, vmin=float(vmin)
                )
            elif vmax:
                data_to_plot.plot.contourf(
                    x="distance", y="depth", cmap=cmap, vmax=float(vmax)
                )
            else:
                data_to_plot.plot.contourf(cmap=cmap)
                # data_to_plot.plot.contourf(x="distance", y="depth", cmap=cmap)
        except Exception as ex:
            message = f"[ERR] Bad data selection: {ex}\n{traceback.print_exc()}"
            logging.error(message)
            return Response(content=create_error_image(message), media_type="image/png")

        # Set the depth limits for display.
        logging.warning(f"\n\n[INFO] Depth limits:{depth}")
        plt.ylim(-depth[1], -depth[0])

        # plt.gca().invert_yaxis()  # Invert the y-axis to show depth increasing downwards
        # Getting current y-axis tick labels
        labels = [item.get_text() for item in plt.gca().get_yticklabels()]
        y_ticks = plt.gca().get_yticks()

        # Assuming the labels are numeric, convert them to float, multiply by -1, and set them back.
        # If labels are not set or are custom, you might need to adjust this part.
        new_labels = [
            custom_formatter(-1.0 * float(label.replace("−", "-"))) if label else 0.0
            for label in labels
        ]  # Handles empty labels as well

        # Setting new labels ( better to explicitly set both the locations of the ticks and their labels using
        # set_yticks along with set_yticklabels.)
        plt.gca().set_yticks(y_ticks)
        plt.gca().set_yticklabels(new_labels)

        plt.ylabel(y_label)
        plt.xlabel(x_label)
        # Calculate section azimuth
        section_azimuth = calculate_azimuth(start[0], start[1], end[0], end[1])

        # Adding the fault locations.
        if label_faults:
            for ind, dist_m in enumerate(fault_distances):
                dist = dist_m
                if units == "mks":
                    dist = dist_m / 1000.0
                plt.text(
                    dist,
                    1.05,
                    f"⟵ {intersection_names[ind]}",
                    rotation=90,
                    transform=plt.gca().get_xaxis_transform(),
                    verticalalignment="bottom",
                    fontsize=7,
                    color="blue",
                )

        # Create the legend text
        legend_text = "\n".join(
            [
                f"{intersection_names[i]}: {intersection_name_info[i]}"
                for i in range(len(intersection_names))
            ]
        )

        # Plotting using the current axis
        fig = plt.gcf()  # Get current figure, if you already have a figure created
        ax = plt.gca()  # Get current axes

        # MB This is a temporary extraction.
        for info_index, info in enumerate(intersection_name_info):
            """
            items = info.split(",")
            dip, dip_dir = items[0].strip().split(":")[1].strip().split("°")
            rake = items[1].split(":")[1]
            lower_depth = items[2].strip().split()[3]
            upper_depth = items[2].strip().split()[1]

            feature_props = {
                "dip": float(dip),
                "dip_dir": dip_dir,
                "rake": float(rake),
                "lower_depth": float(lower_depth),
                "upper_depth": float(upper_depth),
            }

            # Convert dip direction to azimuth
            feature_azimuth = dip_dir_to_azimuth(feature_props["dip_dir"])

            # Project feature onto section
            h_start, h_end, up_depth, low_depth = project_feature(
                feature_props["dip"],
                feature_azimuth,
                section_azimuth,
                feature_props["upper_depth"],
                feature_props["lower_depth"],
                start[0],
                start[1],
            )

            dist_0 = fault_distances[info_index] + h_start
            dist_1 = fault_distances[info_index] + h_end
            depth_0 = up_depth
            depth_1 = low_depth
            if units == "mks":
                dist_0 = (fault_distances[info_index] + h_start) / 1000.0
                dist_1 = (fault_distances[info_index] + h_end) / 1000.0
            else:
                depth_0 = up_depth * 1000
                depth_1 = low_depth * 1000

            # Add white stroke line
            stroke_line = Line2D(
                [
                    dist_0,
                    dist_1,
                ],
                [-1 * depth_0, -1 * depth_1],
                color="white",
                linewidth=4,
                marker="o",
                markerfacecolor="black",
                markeredgewidth=2,
                markeredgecolor="white",
                zorder=100000,
            )
            ax.add_line(stroke_line)

            # Add main fault line
            main_line = Line2D(
                [
                    dist_0,
                    dist_1,
                ],
                [-1 * depth_0, -1 * depth_1],
                color="black",
                linewidth=2,
                marker="o",
                markerfacecolor="black",
                markeredgewidth=1,
                markeredgecolor="white",
                zorder=100001,
            )
            ax.add_line(main_line)
        """
        # Adding vertical text for start and end locations
        plt.text(
            cumulative_distances[0],
            1.05,
            f"⟸{start}",
            rotation=90,
            transform=plt.gca().get_xaxis_transform(),
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=9,
        )
        plt.text(
            cumulative_distances[-1],
            1.05,
            f"⟸{end}",
            rotation=90,
            transform=plt.gca().get_xaxis_transform(),
            verticalalignment="bottom",
            horizontalalignment='center',
            fontsize=9,
        )

        # Add the legend box with the top at the specified position
        plt.text(
            0.5,
            -0.4,
            legend_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="left",  # Align text left
            fontsize=5,  # Smaller font size
            bbox=dict(
                facecolor="white",
                alpha=0.5,
                edgecolor="black",
                boxstyle="round,pad=0.5",
            ),
        )

        # Adjust layout to make space for the legend
        plt.subplots_adjust(bottom=0.3)

        # Setting the aspect ratio to 1:1 ('equal')
        plt.gca().set_aspect(vertical_exaggeration)
        fig.tight_layout()

        # Assuming `plt` is already being used for plotting
        fig = plt.gcf()  # Get the current figure
        axes = fig.axes  # Get all axes objects in the figure

        # The first axes object (`axes[0]`) is typically the main plot
        # The last axes object (`axes[-1]`) is likely the colorbar, especially if it was added automatically
        plot_axes = axes[0]
        colorbar_axes = axes[-1]
        colorbar_axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        # Get the position of the plot's y-axis
        plot_pos = plot_axes.get_position()

        # Adjust the colorbar's position
        # The numbers here are [left, bottom, width, height]
        # You may need to adjust these values based on your specific figure layout
        new_cbar_pos = [plot_pos.x1 + 0.01, plot_pos.y0, 0.02, plot_pos.height]
        colorbar_axes.set_position(new_cbar_pos)

        plt.title(title)

        ax = axes[0]
        # Logo for the plot.
        if os.path.isfile(LOGO_FILE):

            plot_ax = axes[0]
            logo = plt.imread(LOGO_FILE)
            # Get aspect ratio of the logo to maintain proportion
            aspect_ratio = logo.shape[1] / logo.shape[0]  # width / height
            # Get aspect ratio of the logo to maintain proportion
            aspect_ratio = logo.shape[1] / logo.shape[0]  # width / height

            # Desired logo height in inches and its normalized figure coordinates
            logo_height_inches = 0.3  # Height of the logo in inches
            logo_width_inches = (
                logo_height_inches * aspect_ratio
            )  # Calculate width based on aspect ratio

            # Normalize logo size relative to figure size
            norm_logo_width = logo_width_inches / fig_width
            norm_logo_height = logo_height_inches / fig_height

            # Calculate logo position in normalized coordinates
            norm_offset_below = LOGO_INCH_OFFSET_BELOW / fig_height

            # Position of logo's bottom edge
            logo_bottom = (
                plot_ax.get_position().ymin - norm_logo_height - norm_offset_below
            )
            logo_left = plot_ax.get_position().xmin  # Align left edges

            # Create an axes for the logo at the calculated position
            logo_ax = fig.add_axes(
                [logo_left, logo_bottom, norm_logo_width, norm_logo_height]
            )
            logo_ax.imshow(logo)
            logo_ax.axis("off")  # Hide the axis

            # Add a line of text below the logo
            text_y_position = -logo_bottom - norm_logo_height
            logo_ax.text(
                logo_left,
                text_y_position,
                f"{version_label}\n",
                ha="left",
                fontsize=6,
                color="#004F59",
            )

        # Save plot to BytesIO buffer and encode in Base64
        plot_buf = BytesIO()
        plt.savefig(plot_buf, format="png")
        plot_buf.seek(0)
        base64_plot = base64.b64encode(plot_buf.getvalue()).decode("utf-8")
        plt.clf()

    # Convert xarray dataset to CSV (for simplicity in this example)
    # You can also consider other formats like NetCDF for more complex data
    csv_buf = BytesIO()
    plot_data[plot_var].to_dataframe().to_csv(csv_buf)
    csv_buf.seek(0)
    base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

    content = {"image": base64_plot, "csv_data": base64_csv}

    plot_data.close()
    del plot_data
    del data_to_plot
    gc.collect()

    # Return both the plot and data as Base64-encoded strings
    return JSONResponse(content=content)


# Plot an interpolated depth slice.
@router.route("/depth-slice-viewer", methods=["POST"])
async def depthslice(request: Request):
    form_data = await request.form()
    version = "v.2024.102"
    version_label = f"{version}  {utc_now()['date_time']} UTC"

    items_dict = dict(form_data.items())
    logging.info(f"[INFO] POST: {items_dict}")

    try:
        plot_data = xr.load_dataset(
            os.path.join("static", "netcdf", items_dict["data_file"])
        )
        logging.warning(f"\n\n[INFO] POST: {items_dict}")
        start_lat = float(items_dict["start_lat"])
        start_lon = float(items_dict["start_lng"])
        end_lat = float(items_dict["end_lat"])
        end_lon = float(items_dict["end_lng"])
        depth = float(items_dict["start_depth"])
        lon_points = int(items_dict["lng_points"])
        lat_points = int(items_dict["lat_points"])
        image_width = float(items_dict["image_width"])
        image_height = float(items_dict["image_height"])
        figsize = (image_width, image_height)
        dpi = 100

        plot_grid_mapping = items_dict["plot_grid_mapping"]
        units = items_dict["units"]
        title = items_dict["title"]

        unit_standard, depth_factor = unit_conversion_factor(
            plot_data["depth"].attrs["units"], units
        )
        logging.warning(
            f"[INFO] depth units: {plot_data['depth'].attrs['units']} to {unit_standard}, {depth_factor}"
        )
        if depth_factor != 1:
            # Create a new array with modified depth values
            new_depth_values = plot_data["depth"] * depth_factor

            # Create a new coordinate with the new values and the same attributes
            new_depth = xr.DataArray(
                new_depth_values, dims=["depth"], attrs=plot_data["depth"].attrs
            )

            # Update to the new units.
            new_depth.attrs["units"] = unit_standard

            # Update to the new long name.
            new_depth.attrs["long_name"] = (
                f"{new_depth.attrs['standard_name']} [{unit_standard}]"
            )

            # Replace the old 'depth' coordinate with the new one
            plot_data = plot_data.assign_coords(depth=new_depth)

        else:
            # Update to the new units.
            plot_data.depth.attrs["units"] = unit_standard

            # Update to the new long name.
            plot_data.depth.attrs["long_name"] = (
                f"{plot_data.depth.attrs['standard_name']} ({unit_standard})"
            )

        vmin = items_dict["start_value"].strip()
        if vmin == "auto":
            vmin = ""

        vmax = items_dict["end_value"].strip()
        if vmax == "auto":
            vmax = ""

        # We will be working with the variable dataset, so capture the metadata for the main dataset now.
        meta = plot_data.attrs

        # The plot variable.
        plot_var = items_dict["plot_variable"]

        # Interpolate the dataset vertically to the target depth
        plot_data = plot_data[plot_var]
        unit_standard, var_factor = unit_conversion_factor(
            plot_data.attrs["units"], units
        )
        plot_data.data *= var_factor
        logging.warn(f"\n\n{plot_data}\nAttrs: {plot_data.attrs}")
        plot_data.attrs["units"] = unit_standard
        plot_data.attrs["display_name"] = (
            f"{plot_data.attrs['long_name']} [{unit_standard}]"
        )

    except Exception as ex:
        logging.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(
                f"[ERR] Bad selection: \n{ex}\n{traceback.print_exc()}"
            ),
            media_type="image/png",
        )

    utm_zone = None
    logging.warning(f"\n\n[INFO] META: {meta}")
    if "grid_mapping_name" not in meta:
        logging.warning(
            f"[WARN] The 'grid_mapping_name' attribute not found. Assuming geographic coordinate system"
        )
        grid_mapping_name = "latitude_longitude"
    else:
        grid_mapping_name = meta["grid_mapping_name"]

    if "utm_zone" in meta:
        utm_zone = meta["utm_zone"]
        ellipsoid = meta["ellipsoid"]

        logging.info(f"[INFO] UTM zone: {utm_zone}")
        logging.info(f"[INFO] grid_mapping_name: ", grid_mapping_name)

    if grid_mapping_name == "latitude_longitude":
        # Create a 2D grid of latitudes and longitudes
        lon_list = np.linspace(start_lon, end_lon, lon_points).tolist()
        lat_list = np.linspace(start_lat, end_lat, lat_points).tolist()
        x_list, y_list = project_lonlat_utm(lon_list, lat_list, utm_zone, ellipsoid)
        x_grid, y_grid = np.meshgrid(lon_list, lat_list)
        logging.warning(
            f"\n\n[INFO] plot_grid_mapping: {plot_grid_mapping}, grid_mapping_name:{grid_mapping_name}\nlat_list:{lat_list[0]} to {lat_list[-1]}, lon_list: {lon_list[0]} to {lon_list[-1]}"
        )

    else:
        start_x, start_y = project_lonlat_utm(start_lon, start_lat, utm_zone, ellipsoid)
        end_x, end_y = project_lonlat_utm(end_lon, end_lat, utm_zone, ellipsoid)
        x_list = np.linspace(start_x, end_x, lon_points).tolist()
        y_list = np.linspace(start_y, end_y, lat_points).tolist()
        lon_list, lat_list = project_lonlat_utm(
            x_list, y_list, utm_zone, ellipsoid, xy_to_latlon=True
        )
        x_grid, y_grid = np.meshgrid(x_list, y_list)
        logging.warning(
            f"\n\n[INFO]plot_ grid_mapping: {plot_grid_mapping}, grid_mapping_name:{grid_mapping_name}\nx_list:{x_list[0]} to {x_list[-1]}, y_list: {y_list[0]} to {y_list[-1]}"
        )

    # Iterate through the model variables and plot each cross-section.
    cmap = items_dict["colormap"]
    # Cross-section interpolation type.
    interp_type = items_dict["interpolation_method"]
    figsize = (image_width, image_height)
    interpolated_values_2d = np.zeros_like(
        x_grid
    )  # Initialize a 2D array for storing interpolated values

    # No interpolation.
    if interp_type == "none":
        logging.warning(f"[INFO] plot_data: {plot_data}")
        start_lat = closest(plot_data["latitude"].data, start_lat)
        end_lat = closest(plot_data["latitude"].data, end_lat)
        start_lon = closest(plot_data["longitude"].data, start_lon)
        end_lon = closest(plot_data["longitude"].data, end_lon)
        depth_closest = closest(list(plot_data["depth"].data), depth)
        plot_data = plot_data.where(plot_data.depth == depth_closest, drop=True)
        plot_data = plot_data.where(
            (plot_data.latitude >= start_lat)
            & (plot_data.latitude <= end_lat)
            & (plot_data.longitude >= start_lon)
            & (plot_data.longitude <= end_lon),
            drop=True,
        )
        # Creating the contour plot
        # ds_var.plot(figsize=(7, 10), cmap=cmap)#, vmin=vmin, vmax=vmax)
        logging.warning(
            f"\n\n[INFO] x2, y2 ({plot_grid_mapping}<=>{grid_mapping_name}):{grid_mapping_dict[plot_grid_mapping]['x2']}, {grid_mapping_dict[plot_grid_mapping]['y2']}"
        )

        if plot_grid_mapping == grid_mapping_name:
            x2 = grid_mapping_dict[plot_grid_mapping]["x"]
            y2 = grid_mapping_dict[plot_grid_mapping]["y"]
            logging.warning(f"\n\n[INFO] x: {x2}")
        else:
            x2 = grid_mapping_dict[plot_grid_mapping]["x2"]
            y2 = grid_mapping_dict[plot_grid_mapping]["y2"]
            logging.warning(f"\n\n[INFO] x2: {x2}")

        if plot_grid_mapping == "latitude_longitude":
            logging.warning(f"\n\n[INFO] {plot_grid_mapping}")
            unit_standard, x_factor = unit_conversion_factor(
                plot_data[x2].attrs["units"], units
            )
            plot_data[x2].attrs["units"] = unit_standard
            plot_data[x2].attrs[
                "display_name"
            ] = f"{plot_data[x2].attrs['long_name']} [{unit_standard}]"

            unit_standard, y_factor = unit_conversion_factor(
                plot_data[y2].attrs["units"], units
            )
            plot_data[y2].attrs["units"] = unit_standard
            plot_data[y2].attrs[
                "display_name"
            ] = f"{plot_data[y2].attrs['long_name']} [{unit_standard}]"
        else:
            logging.warning(f"\n\n[INFO] {plot_data}")
            unit_standard, x_factor = unit_conversion_factor(
                plot_data[x2].attrs["units"], units
            )
            logging.warning(
                f"\n\n[INFO] {x2} plot_grid_mapping: {plot_grid_mapping}, {unit_standard}, {x_factor}"
            )

            logging.warning(f"\n\n[INFO] plot_data[x2].attrs:{ plot_data[x2].attrs}")
            new_x_values = plot_data[x2] * x_factor
            new_x = xr.DataArray(
                new_x_values, dims=plot_data[x2].dims, attrs=plot_data[x2].attrs
            )
            new_x.attrs["units"] = unit_standard
            new_x.attrs["display_name"] = (
                f"{new_x.attrs['long_name']} [{unit_standard}]"
            )
            logging.warning(f"\n\n[INFO] NEW plot_data[x2].attrs:{ new_x}")

            plot_data = plot_data.assign_coords(x2=new_x)
            logging.warning(f"\n\n[INFO] Updated plot_data[x2].attrs:{ plot_data[x2]}")

            unit_standard, y_factor = unit_conversion_factor(
                plot_data[y2].attrs["units"], units
            )

            new_y_values = plot_data[y2] * y_factor
            new_y = xr.DataArray(
                new_y_values, dims=plot_data[y2].dims, attrs=plot_data[y2].attrs
            )
            new_y.attrs["units"] = unit_standard
            new_y.attrs["display_name"] = (
                f"{new_y.attrs['long_name']} [{unit_standard}]"
            )
            plot_data = plot_data.assign_coords(y2=new_y)

        logging.warning(f"\n\n[INFO] plot_data:{plot_data}")
        # Geographic or projected coordinates?
        if plot_grid_mapping == "latitude_longitude":
            # Calculate the correct aspect ratio
            lat_list = plot_data.latitude.values.flatten()
            lon_list = plot_data.longitude.values.flatten()
            mid_lat = sum(lat_list) / len(lat_list)  # Average latitude
            aspect_ratio = 1 / math.cos(math.radians(mid_lat))
        else:
            aspect_ratio = 1
        x_list = plot_data[x2].values.flatten()
        y_list = plot_data[y2].values.flatten()
        fig_width, fig_height = figsize  # Figure size in inches
        # fig_height = fig_width * aspect_ratio * (max(y_list) - min(y_list)) /  (max(x_list) - min(x_list))
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        logging.warning(
            f"\n\n[INFO] plot_data: {plot_data}\n\n{plot_data.var}\n\n{plot_data.variable}"
        )
        colorbar_label = f"{plot_data.attrs['long_name']} [{standard_units(plot_data.attrs['units'])}]"
        try:
            logging.warning(f"\n\n[INFO] PLOT y2: {y2}\n{plot_data[y2].values}")
            # Creating the contour plot
            # fig = plt.figure(figsize=figsize)
            if plot_grid_mapping == "latitude_longitude":
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

                if vmin and vmax:
                    vmin = min(
                        float(items_dict["start_value"]), float(items_dict["end_value"])
                    )
                    vmax = max(
                        float(items_dict["start_value"]), float(items_dict["end_value"])
                    )

                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                    )
                elif vmin:
                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        vmin=float(vmin),
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                    )
                elif vmax:
                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        vmax=float(vmax),
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                    )
                else:
                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                    )

                plt.xticks(rotation=45)  # Rotating the x-axis labels to 45 degrees
                # Add coastlines
                ax.coastlines()

                # Optionally, add other geographic features
                ax.add_feature(cfeature.BORDERS, linestyle=":")

                # Add gridlines with labels only on the left and bottom
                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False  # Disable top labels
                gl.right_labels = False  # Disable right labels
            else:
                if vmin and vmax:
                    vmin = min(
                        float(items_dict["start_value"]), float(items_dict["end_value"])
                    )
                    vmax = max(
                        float(items_dict["start_value"]), float(items_dict["end_value"])
                    )

                    plot_data.plot(
                        figsize=figsize, cmap=cmap, vmin=vmin, vmax=vmax, x=x2, y=y2
                    )
                elif vmin:
                    plot_data.plot(
                        figsize=figsize, cmap=cmap, vmin=float(vmin), x=x2, y=y2
                    )
                elif vmax:
                    plot_data.plot(
                        figsize=figsize, cmap=cmap, vmax=float(vmax), x=x2, y=y2
                    )
                else:
                    plot_data.plot(figsize=figsize, cmap=cmap, x=x2, y=y2)

                plt.xticks(rotation=45)  # Rotating the x-axis labels to 45 degrees

        except Exception as ex:
            message = f"[ERR] Bad data selection: {ex}\n{traceback.print_exc()}"
            logging.error(message)
            return Response(content=create_error_image(message), media_type="image/png")
    # Interpolation.
    else:
        logging.info(
            f"[INFO] grid_mapping_name: {grid_mapping_name}, plot_grid_mapping: {plot_grid_mapping}"
        )

        # If you're working with longitude and latitude
        if grid_mapping_name == "latitude_longitude":
            interpolated_values_2d = plot_data.interp(
                latitude=lat_list, longitude=lon_list, depth=depth, method=interp_type
            )
        else:
            interpolated_values_2d = plot_data.interp(
                y=y_list, x=x_list, depth=depth, method=interp_type
            )
        # Creating the contour plot
        # fig = plt.figure(figsize=figsize)

        # Check if all elements are NaN. No cubic spline with arrays with NaN.
        if np.all(np.isnan(interpolated_values_2d)) and interp_type == "cubic":
            message = (
                f"[ERR] Data with NaN values. Can't use the cubic spline interpolation"
            )
            logging.error(message)
            base64_plot = base64.b64encode(create_error_image(message)).decode("utf-8")
            plt.clf()

            # Convert xarray dataset to CSV (for simplicity in this example)
            # You can also consider other formats like NetCDF for more complex data
            csv_buf = BytesIO()
            plot_data.to_dataframe().to_csv(csv_buf)
            csv_buf.seek(0)
            base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

            content = {"image": base64_plot, "csv_data": base64_csv}

            plot_data.close()
            del plot_data
            gc.collect()

            # Return both the plot and data as Base64-encoded strings
            return JSONResponse(content=content)

        logging.info(f"[INFO] vmin, vmax (1): ", vmin, vmax)

        if grid_mapping_name == plot_grid_mapping:
            x_label = plot_data[grid_mapping_dict[plot_grid_mapping]["x"]].attrs[
                "long_name"
            ]
            y_label = plot_data[grid_mapping_dict[plot_grid_mapping]["y"]].attrs[
                "long_name"
            ]
        else:
            x_label = plot_data[grid_mapping_dict[plot_grid_mapping]["x2"]].attrs[
                "long_name"
            ]
            y_label = plot_data[grid_mapping_dict[plot_grid_mapping]["y2"]].attrs[
                "long_name"
            ]
        # Geographic or projected coordinates?
        if plot_grid_mapping == "latitude_longitude":
            # Calculate the correct aspect ratio
            _list = plot_data.latitude.values.flatten()
            lon_range = max(plot_data.longitude.values.flatten()) - min(
                plot_data.longitude.values.flatten()
            )
            mid_lat = sum(_list) / len(_list)  # Average latitude
            aspect_ratio = 1 / math.cos(math.radians(mid_lat))
            fig_width, fig_height = figsize  # Figure size in inches
            # fig_height = fig_width * aspect_ratio * (max(_list) - min(_list)) / lon_range

        else:
            aspect_ratio = 1
            fig_width, fig_height = figsize  # Figure size in inches
            # fig_height = fig_width * aspect_ratio * (max(y_list) - min(y_list)) /  (max(x_list) - min(x_list))

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        if plot_grid_mapping == "latitude_longitude":
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        logging.warning(f"\n\n[INFO figsize = {fig_width}, {fig_height}")

        try:
            if vmin and vmax:
                vmin = min(
                    float(items_dict["start_value"]), float(items_dict["end_value"])
                )
                vmax = max(
                    float(items_dict["start_value"]), float(items_dict["end_value"])
                )
                # plt.contourf(lon_list, lat_list, interpolated_values_2d, levels=15, cmap=cmap,vmin=vmin, vmax=vmax)
                # ds_var.plot(figsize=figsize, cmap=cmap,vmin=vmin, vmax=vmax)
            if not vmin:
                vmin = nanmin(interpolated_values_2d)
                # plt.contourf(lon_list, lat_list, interpolated_values_2d, levels=15, cmap=cmap,vmin=vmin)
            if not vmax:
                vmax = nanmax(interpolated_values_2d)
                # plt.contourf(lon_list, lat_list, interpolated_values_2d, levels=15, cmap=cmap, vmax=vmax)
            logging.info(f"[INFO] vmin, vmax (2): ", vmin, vmax)
            levels = np.linspace(vmin, vmax, 15)

            logging.warning(f"\n\n[INFO] PLOT: plot_grid_mapping: {plot_grid_mapping}")
            if plot_grid_mapping == "latitude_longitude":
                # contourf = plt.contourf(lon_list, lat_list, interpolated_values_2d, levels=levels, cmap=cmap)#, x=grid_mapping_dict[grid_mapping]["x"], y=grid_mapping_dict[grid_mapping]["y"])
                # Plot the contourf with longitude and latitude lists
                contourf = ax.contourf(
                    lon_list,
                    lat_list,
                    interpolated_values_2d,
                    levels=levels,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                )

                # Add coastlines
                ax.coastlines()

                # Optionally, add other features like borders and gridlines
                ax.add_feature(cfeature.BORDERS)
                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False  # Disable top labels
                gl.right_labels = False  # Disable right labels
                # logging.warning(f"\n\n[INFO] lon_list:{lon_list}\nlat_list:{lat_list}")

            else:
                logging.warning(f"\n\n[INFO] y_list: {y_list}")
                contourf = plt.contourf(
                    x_list, y_list, interpolated_values_2d, levels=levels, cmap=cmap
                )  # , x=grid_mapping_dict[grid_mapping]["x"], y=grid_mapping_dict[grid_mapping]["y"])
                logging.warning(f"\n\n[INFO] x_list:{x_list}\ny_list:{y_list}")
            logging.warning(f"\n\n[INFO] Plot Limits = {plt.xlim()}, {plt.ylim()}")
            # plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        except Exception as ex:
            message = f"[ERR] Bad data selection: {ex}\n{traceback.print_exc()}"
            logging.error(message)
            return Response(content=create_error_image(message), media_type="image/png")
        # plt.contourf(lon_list, lat_list, interpolated_values_2d, levels=15, cmap=cmap)
        # plt.colorbar(label='Interpolated Value')
        cbar = plt.colorbar(contourf, ax=plt.gca())
        cbar.set_label(
            plot_data.attrs["display_name"]
        )  # Optionally add a label to the colorbar
    plt.xticks(rotation=45)  # Rotating the x-axis labels to 45 degrees

    # Get the dimensions of the Axes object in inches
    bbox = plt.gca().get_position()
    width, height = bbox.width * fig.get_figwidth(), bbox.height * fig.get_figheight()

    logging.warning(f"[INFO] Axes width in inches: {width}")
    logging.warning(f"[INFO] Axes height in inches: {height}")

    # fig.tight_layout()

    # Assuming `plt` is already being used for plotting
    fig = plt.gcf()  # Get the current figure
    axes = fig.axes  # Get all axes objects in the figure

    # The first axes object (`axes[0]`) is typically the main plot
    # The last axes object (`axes[-1]`) is likely the colorbar, especially if it was added automatically
    plot_axes = axes[0]
    colorbar_axes = axes[-1]
    # logging.warning(f"\n\n[INFO] colorbar_label: {colorbar_label}")
    # logging.warning(f"\n\n[INFO] colorbar_axes: {dir(colorbar_axes)}")
    # colorbar_axes.set_label(colorbar_label)
    # colorbar_axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Get the position of the plot's y-axis
    plot_pos = plot_axes.get_position()

    # Adjust the colorbar's position
    # The numbers here are [left, bottom, width, height]
    # You may need to adjust these values based on your specific figure layout
    new_cbar_pos = [plot_pos.x1 + 0.01, plot_pos.y0, 0.02, plot_pos.height]
    colorbar_axes.set_position(new_cbar_pos)
    colorbar_axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    plt.title(title)

    ax = axes[0]
    ax.set_aspect(aspect_ratio)  # Set the aspect ratio to the calculated value
    # Logo for the plot.
    if os.path.isfile(LOGO_FILE):

        plot_ax = axes[0]
        logo = plt.imread(LOGO_FILE)
        # Get aspect ratio of the logo to maintain proportion
        aspect_ratio = logo.shape[1] / logo.shape[0]  # width / height
        # Get aspect ratio of the logo to maintain proportion
        aspect_ratio = logo.shape[1] / logo.shape[0]  # width / height

        # Desired logo height in inches and its normalized figure coordinates
        logo_height_inches = 0.3  # Height of the logo in inches
        logo_width_inches = (
            logo_height_inches * aspect_ratio
        )  # Calculate width based on aspect ratio

        # Normalize logo size relative to figure size
        norm_logo_width = logo_width_inches / fig_width
        norm_logo_height = logo_height_inches / fig_height

        # Calculate logo position in normalized coordinates
        norm_offset_below = 2 * LOGO_INCH_OFFSET_BELOW / fig_height

        # Position of logo's bottom edge
        logo_bottom = plot_ax.get_position().ymin - norm_logo_height - norm_offset_below
        logo_left = plot_ax.get_position().xmin  # Align left edges

        # Create an axes for the logo at the calculated position
        logo_ax = fig.add_axes(
            [logo_left, logo_bottom, norm_logo_width, norm_logo_height]
        )
        logo_ax.imshow(logo)
        logo_ax.axis("off")  # Hide the axis

        # Add a line of text below the logo
        text_y_position = -logo_bottom - norm_logo_height
        logo_ax.text(
            logo_left,
            text_y_position,
            f"{version_label}\n",
            ha="left",
            fontsize=6,
            color="#004F59",
        )

    else:
        logging.warning(f"[WARN] Logo file not found: {LOGO_FILE}")
    # Save the plot to a BytesIO object
    # plot_bytes = BytesIO()
    # plt.savefig(plot_bytes, format='png')
    # plot_bytes.seek(0)
    # plt.clf()
    # del ds_var
    # del ds
    # gc.collect()

    # Return the image as a response
    # return Response(content=plot_bytes.getvalue(), media_type="image/png")

    # Save plot to BytesIO buffer and encode in Base64
    plot_buf = BytesIO()
    plt.savefig(plot_buf, format="png")
    plot_buf.seek(0)
    base64_plot = base64.b64encode(plot_buf.getvalue()).decode("utf-8")
    plt.clf()
    # Convert xarray dataset to CSV (for simplicity in this example)
    # You can also consider other formats like NetCDF for more complex data
    """
    csv_buf = BytesIO()
    plot_data.to_dataframe().to_csv(csv_buf)
    csv_buf.seek(0)
    base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")
    content = {"image": base64_plot, "csv_data": base64_csv}
    """
    try:
        # Check if data is too large, for example by number of elements
        if plot_data.size > 1000000:  # Adjust this size limit as necessary
            raise ValueError("Data too large")

        # Convert data to DataFrame and then to CSV
        csv_buf = BytesIO()
        plot_data.to_dataframe().to_csv(csv_buf)
        csv_buf.seek(0)
        base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

    except ValueError as e:
        # Handle cases where data is too large
        csv_buf = BytesIO()
        # Write a message indicating the data is too large
        csv_buf.write(b"Data too large")
        csv_buf.seek(0)
        base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

    except Exception as e:
        # Handle other potential exceptions
        csv_buf = BytesIO()
        error_message = f"An error occurred: {str(e)}"
        csv_buf.write(error_message.encode())
        csv_buf.seek(0)
        base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

    # Package the content
    content = {"image": base64_plot, "csv_data": base64_csv}

    plot_data.close()
    del plot_data
    gc.collect()

    # Return both the plot and data as Base64-encoded strings
    return JSONResponse(content=content)
