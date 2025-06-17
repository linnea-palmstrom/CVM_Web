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

import numpy as np

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


def get_geocsv_metadata_from_ds(ds):
    """Compose GeoCSV style metadata for a given Dataset.

    Keyword arguments:
    ds -- [required] the xarray dataset
    """
    geocsv_metadata = list()
    for row in ds.attrs:
        # For some reason history lines are split (investigate).
        if row == "history":
            ds.attrs[row] = ds.attrs[row].replace("\n", ";")
        geocsv_metadata.append(f"# global_{row}: {ds.attrs[row]}")
    for var in ds.variables:
        if "variable" not in ds[var].attrs:
            geocsv_metadata.append(f"# {var}_variable: {var}")
            geocsv_metadata.append(f"# {var}_dimensions: {len(ds[var].dims)}")

        for att in ds[var].attrs:
            geocsv_metadata.append(f"# {var}_{att}: {ds[var].attrs[att]}")
            if att == "missing_value":
                geocsv_metadata.append(f"# {var}_missing_value: nan")
            if att == "variable":
                geocsv_metadata.append(f"# {var}_dimensions: {len(ds[var].dims)}")
                geocsv_metadata.append(f"# {var}_column: {var}")
    metadata = "\n".join(geocsv_metadata)
    metadata = f"{metadata}\n"
    return metadata


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


def subsetter(ds, limits):
    """
    Subset a dataset as a volume.

    Call arguments:
        ds - [required] the xarray dataset
        limits - [required] limits of the volume in all directions.
    """
    geospatial_dict = {
        "latitude": ["geospatial_lat_min", "geospatial_lat_max"],
        "longitude": ["geospatial_lon_min", "geospatial_lon_max"],
        "depth": ["geospatial_vertical_min", "geospatial_vertical_max"],
    }
    # Check if the array has any zero-sized dimensions
    warnings = ""
    try:
        limit_keys = list(limits.keys())
        limit_values = list(limits.values())
        sliced_data = ds.where(
            (ds[limit_keys[0]] >= limit_values[0][0])
            & (ds[limit_keys[0]] <= limit_values[0][1])
            & (ds[limit_keys[1]] >= limit_values[1][0])
            & (ds[limit_keys[1]] <= limit_values[1][1])
            & (ds[limit_keys[2]] >= limit_values[2][0])
            & (ds[limit_keys[2]] <= limit_values[2][1]),
            drop=True,
        )

        for dim in limit_keys:
            if dim in geospatial_dict:
                #  The dropna method is used to remove coordinates with all NaN values along the specified dimensions
                sliced_data = sliced_data.dropna(dim=dim, how="all")
                if geospatial_dict[dim][0] in sliced_data.attrs:
                    sliced_data.attrs[geospatial_dict[dim][0]] = min(
                        sliced_data[dim].values
                    )
                if geospatial_dict[dim][1] in sliced_data.attrs:
                    sliced_data.attrs[geospatial_dict[dim][1]] = max(
                        sliced_data[dim].values
                    )
    except Exception as ex:
        warnings = ex
        return ds, warnings

    return sliced_data, warnings


@router.get("/drag", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("drag_test.html", {"request": request})


@router.get("/volume-data", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("volume-data.html", {"request": request})


@router.get("/3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("view3d_volume.html", {"request": request})


@router.get("/3d-slice", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("view3d_slice.html", {"request": request})


@router.get("/depth-slice-3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("depth-slice-3d.html", {"request": request})


@router.get("/depth-slice-data", response_class=HTMLResponse)
async def read_page1(request: Request):
    return templates.TemplateResponse("depth-slice-data.html", {"request": request})


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
        dropdown_html += (
            f'<option value="{filename}"{selected}>{model_list[i]}</option>'
        )
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


@router.get("/extract-volume-data")
async def volume_data(
    request: Request,
    data_file: str,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    start_depth: float,
    end_depth: float,
    variables_hidden: str,  # Multiple comma-separated values
    output_format: str = Query(
        "csv"
    ),  # Add output_format argument with default value as "csv"
    interpolation_method: str = Query("none"),
):
    version = "v.2024.102"
    interpolation_method == "none"

    try:
        variables_to_keep = variables_hidden.split(",")
        data_file = urllib.parse.unquote(data_file)  # Decode the URL-safe string
        with xr.open_dataset(os.path.join(NETCDF_DIR, data_file)) as ds:
            subset_limits = dict()
            subset_limits["longitude"] = [start_lng, end_lng]
            subset_limits["latitude"] = [start_lat, end_lat]
            subset_limits["depth"] = [start_depth, end_depth]
            subset_volume, warnings = subsetter(ds, subset_limits)
            output_data = subset_volume[variables_to_keep]
            meta = output_data.attrs
            logging.warning(f"output_data: {output_data}")
            unique_id = uuid4().hex
            if output_format == "csv":
                df = output_data.to_dataframe().reset_index()
                csv_path = f"/tmp/volume_data_{unique_id}.csv"
                df.to_csv(csv_path, index=False)
                return FileResponse(
                    path=csv_path,
                    filename=f"volume_data_{unique_id}.csv",
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=volume_data_{unique_id}.csv"
                    },
                )
            elif output_format == "netcdf":
                netcdf_path = f"/tmp/volume_data_{unique_id}.nc"
                output_data.to_netcdf(netcdf_path)
                return FileResponse(
                    path=netcdf_path,
                    filename=f"volume_data_{unique_id}.nc",
                    media_type="application/netcdf",
                    headers={
                        "Content-Disposition": f"attachment; filename=volume_data_{unique_id}.nc"
                    },
                )
            elif output_format == "geocsv":
                metadata = get_geocsv_metadata_from_ds(output_data)
                df = output_data.to_dataframe().reset_index()
                geocsv_path = f"/tmp/plot_data_{unique_id}.csv"
                with open(geocsv_path, "w") as fp:
                    fp.write(f"{metadata}\n")
                df.to_csv(geocsv_path, index=False, mode="a")
                return FileResponse(
                    path=geocsv_path,
                    filename=f"slice_data_{unique_id}.geocsv",
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=volume_data_{unique_id}.geocsv"
                    },
                )
            else:
                raise HTTPException(
                    status_code=400, detail="Invalid output format specified"
                )

    except Exception as ex:
        logging.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(f"[ERR] {ex}\n{traceback.print_exc()}"),
            media_type="image/png",
        )


@router.get("/slice-data")
async def slice_data(
    request: Request,
    data_file: str,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    start_depth: float,
    lng_points: int,
    lat_points: int,
    units: str,
    variables_hidden: str,  # Multiple comma-separated values
    output_format: str = Query(
        "csv"
    ),  # Add output_format argument with default value as "csv"
    interpolation_method: str = Query("none"),
):
    version = "v.2024.102"

    try:
        data_file = urllib.parse.unquote(data_file)  # Decode the URL-safe string
        output_data = xr.load_dataset(os.path.join("static", "netcdf", data_file))
        logging.info(f"[INFO] GET: {locals()}")

        unit_standard, depth_factor = unit_conversion_factor(
            output_data["depth"].attrs["units"], units
        )
        if depth_factor != 1:
            new_depth_values = output_data["depth"] * depth_factor
            new_depth = xr.DataArray(
                new_depth_values, dims=["depth"], attrs=output_data["depth"].attrs
            )
            new_depth.attrs["units"] = unit_standard
            new_depth.attrs["long_name"] = (
                f"{new_depth.attrs['standard_name']} [{unit_standard}]"
            )
            output_data = output_data.assign_coords(depth=new_depth)
        else:
            output_data.depth.attrs["units"] = unit_standard
            output_data.depth.attrs["long_name"] = (
                f"{output_data.depth.attrs['standard_name']} ({unit_standard})"
            )

        meta = output_data.attrs

        variable_list = variables_hidden.split(",")  # Split the variables by comma
        selected_data_vars = output_data[variable_list]

        # Apply unit conversion to each variable
        for var in variable_list:
            unit_standard, var_factor = unit_conversion_factor(
                selected_data_vars[var].attrs["units"], units
            )
            selected_data_vars[var].data *= var_factor
            selected_data_vars[var].attrs["units"] = unit_standard
            selected_data_vars[var].attrs[
                "display_name"
            ] = f"{selected_data_vars[var].attrs['long_name']} [{unit_standard}]"

    except Exception as ex:
        logging.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(
                f"[ERR] Bad selection: \n{ex}\n{traceback.print_exc()}"
            ),
            media_type="image/png",
        )

    try:
        grid_mapping_name = meta.get("output_grid_mapping", "latitude_longitude")
        utm_zone = meta.get("utm_zone")
        ellipsoid = meta.get("ellipsoid")

        if grid_mapping_name == "latitude_longitude":
            lon_list = np.linspace(start_lng, end_lng, lng_points).tolist()
            lat_list = np.linspace(start_lat, end_lat, lat_points).tolist()
            x_list, y_list = project_lonlat_utm(lon_list, lat_list, utm_zone, ellipsoid)
            x_grid, y_grid = np.meshgrid(lon_list, lat_list)
        else:
            start_x, start_y = project_lonlat_utm(
                start_lng, start_lat, utm_zone, ellipsoid
            )
            end_x, end_y = project_lonlat_utm(end_lng, end_lat, utm_zone, ellipsoid)
            x_list = np.linspace(start_x, end_x, lng_points).tolist()
            y_list = np.linspace(start_y, end_y, lat_points).tolist()
            lon_list, lat_list = project_lonlat_utm(
                x_list, y_list, utm_zone, ellipsoid, xy_to_latlon=True
            )
            x_grid, y_grid = np.meshgrid(x_list, y_list)

        interp_type = interpolation_method

        if interp_type == "none":
            start_lat = closest(output_data["latitude"].data, start_lat)
            end_lat = closest(output_data["latitude"].data, end_lat)
            start_lng = closest(output_data["longitude"].data, start_lng)
            end_lng = closest(output_data["longitude"].data, end_lng)
            depth_closest = closest(list(output_data["depth"].data), start_depth)
            selected_data_vars = selected_data_vars.where(
                output_data.depth == depth_closest, drop=True
            )
            selected_data_vars = selected_data_vars.where(
                (output_data.latitude >= start_lat)
                & (output_data.latitude <= end_lat)
                & (output_data.longitude >= start_lng)
                & (output_data.longitude <= end_lng),
                drop=True,
            )
            data_to_return = selected_data_vars
        else:
            interpolated_values_2d = {}
            for var in variable_list:
                if grid_mapping_name == "latitude_longitude":
                    interpolated_values_2d[var] = selected_data_vars[var].interp(
                        latitude=lat_list,
                        longitude=lon_list,
                        depth=start_depth,
                        method=interp_type,
                    )
                else:
                    interpolated_values_2d[var] = selected_data_vars[var].interp(
                        y=y_list, x=x_list, depth=start_depth, method=interp_type
                    )

            if any(
                np.all(np.isnan(interpolated_values_2d[var])) and interp_type == "cubic"
                for var in variable_list
            ):
                message = "[ERR] Data with NaN values. Can't use the cubic spline interpolation"
                logging.error(message)
                base64_plot = base64.b64encode(create_error_image(message)).decode(
                    "utf-8"
                )
                plt.clf()
                csv_buf = BytesIO()
                selected_data_vars.to_dataframe().to_csv(csv_buf)
                csv_buf.seek(0)
                base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")
                content = {"image": base64_plot, "csv_data": base64_csv}
                output_data.close()
                del output_data
                gc.collect()
                return JSONResponse(content=content)
            data_to_return = xr.Dataset(interpolated_values_2d)

        unique_id = uuid4().hex
        if output_format == "csv":
            df = data_to_return.to_dataframe().reset_index()
            csv_path = f"/tmp/plot_data_{unique_id}.csv"
            df.to_csv(csv_path, index=False)
            return FileResponse(
                path=csv_path,
                filename="slice_data.csv",
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=slice_data.csv"},
            )
        elif output_format == "xarray":
            response_json = jsonable_encoder(data_to_return.to_dict())
            return JSONResponse(content=response_json)
        elif output_format == "netcdf":
            netcdf_path = f"/tmp/slice_data_{unique_id}.nc"
            data_to_return.to_netcdf(netcdf_path)
            return FileResponse(
                path=netcdf_path,
                filename="slice_data.nc",
                media_type="application/netcdf",
                headers={"Content-Disposition": f"attachment; filename=slice_data.nc"},
            )
        elif output_format == "geocsv":
            metadata = get_geocsv_metadata_from_ds(data_to_return)
            df = data_to_return.to_dataframe().reset_index()
            geocsv_path = f"/tmp/plot_data_{unique_id}.csv"
            with open(geocsv_path, "w") as fp:
                fp.write(f"{metadata}\n")
            df.to_csv(geocsv_path, index=False, mode="a")
            return FileResponse(
                path=geocsv_path,
                filename="slice_data.geocsv",
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=slice_data.geocsv"
                },
            )
        else:
            raise HTTPException(
                status_code=400, detail="Invalid output format specified"
            )

    except Exception as ex:
        logging.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(f"[ERR] {ex}\n{traceback.print_exc()}"),
            media_type="image/png",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
