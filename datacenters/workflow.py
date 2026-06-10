from __future__ import annotations

import io
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import niquests
import numpy as np
import pandas as pd
import pyproj
import rasterio
from google.cloud.storage import Client as StorageClient
from obstore.store import LocalStore, ObjectStore, S3Store
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, mapping
from tilebox.datasets import Client as DatasetClient
from tilebox.workflows import ExecutionContext, Runner, Task
from tilebox.workflows.cache import GoogleStorageCache, JobCache, LocalFileSystemCache

DEFAULT_SITES_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/1JJ6kcVo-NjlAYtznwHOki2DVl4WWV6lhy-eXhFCdKKU/"
    "export?format=csv&gid=386766486"
)
DEFAULT_STATUS_FILTER = ["Approved/Permitted/Under construction", "Expanding", "Proposed"]
DEFAULT_GCS_CACHE_PROJECT = "tilebox-hosted-compute"
DEFAULT_GCS_CACHE_BUCKET = "tilebox-hosted-compute-us-central1-results"
DEFAULT_GCS_CACHE_PREFIX = "jobs"

SENTINEL2_COLLECTIONS = ["S2A_S2MSI2A", "S2B_S2MSI2A", "S2C_S2MSI2A"]
BAND_NAMES = ["B02", "B03", "B04", "B08", "B11", "B12"]
CLAY_BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
ALL_BAND_NAMES = sorted(set(BAND_NAMES) | set(CLAY_BAND_NAMES))
BAD_CLOUD_SCL_CLASSES = {3, 8, 9, 10}
INVALID_SCL_CLASSES = {0, 1}
SENTINEL2_10M_RESOLUTION_M = 10.0
MAX_CROP_INVALID_PERCENT = 1.0
EPSILON = 1e-6
CLAY_CHECKPOINT_URL = "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
CLAY_CHECKPOINT_PATH = Path("~/.cache/tilebox/models/clay-v1.5.ckpt").expanduser()
CLAY_MIN_CHECKPOINT_BYTES = 100_000_000
CLAY_METADATA_PATH = Path(__file__).with_name("clay_metadata.yaml")
CLAY_PLATFORM = "sentinel-2-l2a"
CLAY_INPUT_SIZE = 256
CLAY_PATCH_SIZE = 8
CLAY_EMBEDDING_DIM = 1024

JP2_BAND_ASSET_SUFFIXES = {
    "B02": ("B02_10m.jp2",),
    "B03": ("B03_10m.jp2",),
    "B04": ("B04_10m.jp2",),
    "B05": ("B05_20m.jp2",),
    "B06": ("B06_20m.jp2",),
    "B07": ("B07_20m.jp2",),
    "B08": ("B08_10m.jp2",),
    "B8A": ("B8A_20m.jp2",),
    "B11": ("B11_20m.jp2",),
    "B12": ("B12_20m.jp2",),
    "SCL": ("SCL_20m.jp2",),
}


@dataclass(frozen=True)
class Site:
    site_id: str
    name: str
    latitude: float
    longitude: float
    source_ids: list[str]
    operators: list[str]
    source_count: int


@dataclass(frozen=True)
class SceneMetadata:
    status: str
    site_id: str
    label: str
    scene_id: str | None = None
    stac_item_id: str | None = None
    acquisition_time: str | None = None
    crop_cloud_cover: float | None = None
    crop_invalid_percent: float | None = None
    scene_cloud_cover: float | None = None
    bands_key: str | None = None
    preview_key: str | None = None
    data_location: str | None = None
    asset_format: str | None = None
    message: str | None = None


@lru_cache
def sentinel2_data_store() -> ObjectStore:
    eodata_mounted = Path("/eodata")
    if eodata_mounted.exists():
        return LocalStore(eodata_mounted)

    access_key = os.environ.get("COPERNICUS_ACCESS_KEY")
    secret_key = os.environ.get("COPERNICUS_SECRET_KEY")
    if access_key is None or secret_key is None:
        raise ValueError("COPERNICUS_ACCESS_KEY and COPERNICUS_SECRET_KEY must be set")

    endpoint = os.environ.get("COPERNICUS_S3_ENDPOINT", "https://eodata.dataspace.copernicus.eu")
    return S3Store(
        bucket="eodata",
        endpoint=endpoint,
        access_key_id=access_key,
        secret_access_key=secret_key,
    )


def workflow_cache() -> JobCache:
    cache_url = os.environ.get(
        "WORKFLOW_CACHE_BUCKET",
        f"gs://{DEFAULT_GCS_CACHE_BUCKET}/{DEFAULT_GCS_CACHE_PREFIX}",
    )
    if cache_url == "":
        return LocalFileSystemCache("cache")
    if not cache_url.startswith("gs://"):
        raise ValueError(f"Expected WORKFLOW_CACHE_BUCKET to be a gs:// URL, got {cache_url!r}")

    bucket_and_prefix = cache_url.removeprefix("gs://").split("/", 1)
    bucket_name = bucket_and_prefix[0]
    prefix = bucket_and_prefix[1] if len(bucket_and_prefix) == 2 else "jobs"
    project = os.environ.get("WORKFLOW_CACHE_GCP_PROJECT", DEFAULT_GCS_CACHE_PROJECT)
    bucket = StorageClient(project=project).bucket(bucket_name)
    return GoogleStorageCache(bucket, prefix=prefix)


def _json_dumps(data: Any) -> bytes:
    return json.dumps(data, indent=2, sort_keys=True).encode()


def _json_loads(data: bytes) -> Any:
    return json.loads(data.decode())


def _sites_by_id(raw_sites: bytes) -> dict[str, Site]:
    return {item["site_id"]: Site(**item) for item in _json_loads(raw_sites)}


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _date_window(center: str, window_days: int) -> tuple[str, str]:
    center_date = _parse_date(center)
    half_window = window_days // 2
    start = center_date - timedelta(days=half_window)
    end = center_date + timedelta(days=window_days - half_window)
    return start.isoformat(), end.isoformat()


def _utm_crs_for(latitude: float, longitude: float) -> pyproj.CRS:
    zone = int((longitude + 180) // 6) + 1
    epsg = 32600 + zone if latitude >= 0 else 32700 + zone
    return pyproj.CRS.from_epsg(epsg)


def _site_crop_polygon(latitude: float, longitude: float, crop_size_m: int) -> Polygon:
    wgs84 = pyproj.CRS.from_epsg(4326)
    utm = _utm_crs_for(latitude, longitude)
    to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True)
    x, y = to_utm.transform(longitude, latitude)
    half = crop_size_m / 2
    corners = [
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half),
        (x - half, y - half),
    ]
    return Polygon([to_wgs84.transform(px, py) for px, py in corners])


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _first_column(columns: list[str], candidates: list[str]) -> str:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise ValueError(f"CSV is missing any of these columns: {candidates}")


def _download_sites_csv(csv_url: str) -> pd.DataFrame:
    response = niquests.get(csv_url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def _merge_sites(  # noqa: C901
    csv_url: str,
    max_sites: int | None,
    random_seed: int,
    status_filter: list[str],
) -> list[Site]:
    frame = _download_sites_csv(csv_url)
    columns = list(frame.columns)
    lat_col = _first_column(columns, ["lat", "latitude"])
    lon_col = _first_column(columns, ["lon", "long", "longitude", "lng"])
    name_col = _first_column(columns, ["facility_name", "name", "site_name"])
    status_col = _first_column(columns, ["status"])
    operator_col = next((column for column in columns if column.lower() in {"operator", "operator_name"}), None)
    normalized_status_filter = {status.casefold().strip() for status in status_filter}

    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        status = str(row.get(status_col) or "").strip()
        if status.casefold() not in normalized_status_filter:
            continue
        latitude = pd.to_numeric(row[lat_col], errors="coerce")
        longitude = pd.to_numeric(row[lon_col], errors="coerce")
        if pd.isna(latitude) or pd.isna(longitude):
            continue
        name = str(row.get(name_col) or f"site-{index}").strip()
        operator = ""
        if operator_col is not None and not pd.isna(row.get(operator_col)):
            operator = str(row[operator_col]).strip()
        rows.append(
            {
                "source_id": str(index),
                "name": name,
                "operator": operator,
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        )

    parent = list(range(len(rows)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            if _haversine_m(left["latitude"], left["longitude"], right["latitude"], right["longitude"]) <= 1000:
                union(left_index, right_index)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)

    sites: list[Site] = []
    for site_number, group in enumerate(groups.values(), start=1):
        latitude = sum(item["latitude"] for item in group) / len(group)
        longitude = sum(item["longitude"] for item in group) / len(group)
        names = [item["name"] for item in group if item["name"]]
        operators = sorted({item["operator"] for item in group if item["operator"]})
        source_ids = [item["source_id"] for item in group]
        site_id = f"site-{site_number:05d}"
        sites.append(
            Site(
                site_id=site_id,
                name=names[0] if names else site_id,
                latitude=latitude,
                longitude=longitude,
                source_ids=source_ids,
                operators=operators,
                source_count=len(group),
            )
        )

    if max_sites is not None and max_sites < len(sites):
        return random.Random(random_seed).sample(sites, max_sites)  # noqa: S311
    return sites


def _dataset_candidates(  # noqa: PLR0913
    latitude: float,
    longitude: float,
    target_date: str,
    window_days: int,
    crop_size_m: int,
    scene_cloud_cover_max: float,
) -> list[dict[str, Any]]:
    start, end = _date_window(target_date, window_days)
    area = _site_crop_polygon(latitude, longitude, crop_size_m)
    data = DatasetClient().dataset("open_data.copernicus.sentinel2_msi").query(
        collections=SENTINEL2_COLLECTIONS,
        temporal_extent=(start, end),
        spatial_extent=area,
        show_progress=False,
    )
    if data.sizes.get("time", 0) == 0:
        return []

    candidates: list[dict[str, Any]] = []
    cloud_covers = data["cloud_cover"].to_numpy()
    times = data["time"].to_numpy()
    granule_names = data["granule_name"].to_numpy()
    geometries = data["geometry"].to_numpy()
    locations = data["location"].to_numpy()
    for index in range(data.sizes["time"]):
        cloud_cover = float(cloud_covers[index])
        if cloud_cover > scene_cloud_cover_max:
            continue
        time_value = pd.Timestamp(times[index]).to_pydatetime()
        candidates.append(
            {
                "time": time_value,
                "granule_name": str(granule_names[index]),
                "location": str(locations[index]).removeprefix("/eodata/"),
                "cloud_cover": cloud_cover,
                "geometry": geometries[index],
            }
        )

    target = datetime.combine(_parse_date(target_date), datetime.min.time())
    candidates.sort(key=lambda item: (abs((item["time"] - target).total_seconds()), -item["time"].timestamp()))
    return candidates


def _find_copernicus_jp2_assets(granule_location: str) -> dict[str, str]:
    jp2_assets: dict[str, str] = {}
    for page in sentinel2_data_store().list(granule_location):
        for obj in page:
            path = obj["path"]
            for band_name, suffixes in JP2_BAND_ASSET_SUFFIXES.items():
                if band_name not in jp2_assets and any(path.endswith(suffix) for suffix in suffixes):
                    jp2_assets[band_name] = path
    return jp2_assets


def _bounds_for_crs(polygon_wgs84: Polygon, crs: Any) -> tuple[float, float, float, float]:
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs: list[float] = []
    ys: list[float] = []
    for lon, lat in polygon_wgs84.exterior.coords:
        x, y = transformer.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def _read_jp2_asset_crop(asset_path: str, polygon_wgs84: Polygon) -> tuple[np.ndarray, Any, Any]:
    eodata_path = Path("/eodata") / asset_path
    if eodata_path.exists():
        with rasterio.open(eodata_path, driver="JP2OpenJPEG") as source:
            window = from_bounds(*_bounds_for_crs(polygon_wgs84, source.crs), transform=source.transform)
            window = window.round_offsets().round_lengths()
            data = source.read(1, window=window, boundless=False)
            return data, source.window_transform(window), source.crs

    buffer = bytes(sentinel2_data_store().get(asset_path).bytes())
    with rasterio.MemoryFile(buffer).open(driver="JP2OpenJPEG") as source:
        window = from_bounds(*_bounds_for_crs(polygon_wgs84, source.crs), transform=source.transform)
        window = window.round_offsets().round_lengths()
        data = source.read(1, window=window, boundless=False)
        return data, source.window_transform(window), source.crs


def _read_crop(
    asset_paths: dict[str, str],
    latitude: float,
    longitude: float,
    crop_size_m: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    polygon_wgs84 = _site_crop_polygon(latitude, longitude, crop_size_m)

    arrays: dict[str, np.ndarray] = {}
    reference_transform = None
    reference_crs = None
    reference_shape = None

    for band_name in ["B04", "B03", "B02", "B08"]:
        data, transform, crs = _read_jp2_asset_crop(asset_paths[band_name], polygon_wgs84)
        arrays[band_name] = data
        if reference_transform is None:
            reference_transform = transform
            reference_crs = crs
            reference_shape = data.shape

    if reference_transform is None or reference_crs is None or reference_shape is None:
        raise ValueError("Could not read reference Sentinel-2 bands")

    for band_name in ["B05", "B06", "B07", "B8A", "B11", "B12", "SCL"]:
        source_data, source_transform, source_crs = _read_jp2_asset_crop(asset_paths[band_name], polygon_wgs84)
        destination = np.empty(reference_shape, dtype=source_data.dtype)
        reproject(
            source_data,
            destination,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            resampling=Resampling.nearest if band_name == "SCL" else Resampling.bilinear,
        )
        arrays[band_name] = destination

    height, width = reference_shape
    west, south, east, north = array_bounds(height, width, reference_transform)
    metadata = {
        "crs": str(reference_crs),
        "transform": list(reference_transform)[:6],
        "height": int(height),
        "width": int(width),
        "bounds": [float(west), float(south), float(east), float(north)],
        "aoi_geojson": mapping(polygon_wgs84),
    }
    return arrays, metadata


def _bad_fraction(scl: np.ndarray) -> float:
    valid = ~np.isin(scl, list(INVALID_SCL_CLASSES))
    if int(valid.sum()) == 0:
        return 1.0
    bad = np.isin(scl, list(BAD_CLOUD_SCL_CLASSES)) & valid
    return float(bad.sum() / valid.sum())


def _expected_crop_pixels(crop_size_m: int) -> int:
    return math.floor(crop_size_m / SENTINEL2_10M_RESOLUTION_M)


def _has_full_crop_size(crop_metadata: dict[str, Any], crop_size_m: int) -> bool:
    expected_pixels = _expected_crop_pixels(crop_size_m)
    return int(crop_metadata["height"]) >= expected_pixels and int(crop_metadata["width"]) >= expected_pixels


def _invalid_data_fraction(arrays: dict[str, np.ndarray]) -> float:
    invalid_scl = np.isin(arrays["SCL"], list(INVALID_SCL_CLASSES))
    zero_reflectance = np.logical_and.reduce([arrays[band_name] == 0 for band_name in ALL_BAND_NAMES])
    invalid = invalid_scl | zero_reflectance
    return float(invalid.sum() / invalid.size)


def _save_npz(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        **{band_name: arrays[band_name] for band_name in ALL_BAND_NAMES},
        SCL=arrays["SCL"],
        metadata=json.dumps(metadata),
    )
    return buffer.getvalue()


def _load_npz(raw: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(io.BytesIO(raw)) as data:
        arrays = {name: data[name] for name in [*ALL_BAND_NAMES, "SCL"]}
        metadata = json.loads(str(data["metadata"]))
    return arrays, metadata


def _preview_png(arrays: dict[str, np.ndarray]) -> bytes:
    rgb = np.stack([arrays["B04"], arrays["B03"], arrays["B02"]], axis=-1).astype(np.float32)
    valid = (~np.isin(arrays["SCL"], list(INVALID_SCL_CLASSES))) & np.any(rgb > 0, axis=-1)
    values = rgb[valid]
    if values.size == 0:
        scaled = np.zeros(rgb.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = low + 1
        display = np.clip((rgb - low) / (high - low), 0, 1)
        display = np.power(display, 1.0 / 1.2)

        luma = 0.2126 * display[..., 0] + 0.7152 * display[..., 1] + 0.0722 * display[..., 2]
        median_luma = float(np.median(luma[valid]))
        if median_luma > 0:
            gain = float(np.clip(0.35 / median_luma, 1.0, 1.2))
            display = np.clip(display * gain, 0, 1)

        display[~valid] = 0
        scaled = (display * 255).astype(np.uint8)
    image = Image.fromarray(scaled, mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _indices(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    b02 = arrays["B02"].astype(np.float32)
    b03 = arrays["B03"].astype(np.float32)
    b04 = arrays["B04"].astype(np.float32)
    b08 = arrays["B08"].astype(np.float32)
    b11 = arrays["B11"].astype(np.float32)
    return {
        "ndbi": (b11 - b08) / (b11 + b08 + EPSILON),
        "bsi": ((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02) + EPSILON),
        "ndvi": (b08 - b04) / (b08 + b04 + EPSILON),
        "mndwi": (b03 - b11) / (b03 + b11 + EPSILON),
        "brightness": (b02 + b03 + b04) / 3.0,
    }


def _component_score(values: np.ndarray, low: float, high: float) -> float:
    if values.size == 0:
        return 0.0
    value = float(np.nanmedian(values))
    return float(np.clip((value - low) / (high - low), 0, 1) * 100)


def _score_scalar(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip((value - low) / (high - low), 0, 1) * 100)


def _safe_percentile(values: np.ndarray, percentile: float, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return float(np.nanpercentile(values, percentile))


def _mad_threshold(values: np.ndarray, minimum: float) -> float:
    if values.size == 0:
        return minimum
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    return max(minimum, median + 3.0 * 1.4826 * mad)


def _pixel_area_m2(metadata: dict[str, Any]) -> float:
    transform = metadata.get("transform") or []
    if len(transform) >= 6:
        a, b, _, d, e, _ = [float(value) for value in transform[:6]]
        area = abs(a * e - b * d)
        if area > 0:
            return area
    return 100.0


def _connected_component_metrics(mask: np.ndarray, pixel_area_m2: float) -> dict[str, float]:
    visited = np.zeros(mask.shape, dtype=bool)
    largest_pixels = 0
    component_count = 0
    height, width = mask.shape

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        component_count += 1
        pixels = 0
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            pixels += 1
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if visited[neighbor_y, neighbor_x] or not mask[neighbor_y, neighbor_x]:
                        continue
                    visited[neighbor_y, neighbor_x] = True
                    stack.append((neighbor_y, neighbor_x))
        largest_pixels = max(largest_pixels, pixels)

    changed_pixels = int(mask.sum())
    hectares_per_pixel = pixel_area_m2 / 10_000.0
    return {
        "changed_area_ha": changed_pixels * hectares_per_pixel,
        "largest_component_area_ha": largest_pixels * hectares_per_pixel,
        "largest_component_fraction": 0.0 if changed_pixels == 0 else largest_pixels / changed_pixels,
        "component_count": float(component_count),
    }


def _spectral_stack(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([arrays[band_name].astype(np.float32) / 10_000.0 for band_name in BAND_NAMES], axis=0)


def _center_crop_arrays(arrays: dict[str, np.ndarray], height: int, width: int) -> dict[str, np.ndarray]:
    cropped: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        y_offset = max(0, (array.shape[0] - height) // 2)
        x_offset = max(0, (array.shape[1] - width) // 2)
        cropped[name] = array[y_offset : y_offset + height, x_offset : x_offset + width]
    return cropped


def _align_common_shape(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[int, int]]:
    height = min(*(array.shape[0] for array in [*before.values(), *after.values()]))
    width = min(*(array.shape[1] for array in [*before.values(), *after.values()]))
    if height <= 0 or width <= 0:
        raise ValueError("Before/after crops do not have a non-empty common shape")
    if all(array.shape == (height, width) for array in [*before.values(), *after.values()]):
        return before, after, (height, width)
    return _center_crop_arrays(before, height, width), _center_crop_arrays(after, height, width), (height, width)


def _robust_grayscale(
    before_image: np.ndarray, after_image: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([before_image[valid], after_image[valid]])
    if values.size == 0:
        return np.zeros_like(before_image, dtype=np.float32), np.zeros_like(after_image, dtype=np.float32)
    low, high = np.nanpercentile(values, [2, 98])
    if high <= low:
        high = low + 1.0
    before_scaled = np.clip((before_image - low) / (high - low), 0, 1).astype(np.float32)
    after_scaled = np.clip((after_image - low) / (high - low), 0, 1).astype(np.float32)
    return before_scaled, after_scaled


def _masked_ssim(before_image: np.ndarray, after_image: np.ndarray, valid: np.ndarray) -> float:
    before_values = before_image[valid].astype(np.float64)
    after_values = after_image[valid].astype(np.float64)
    if before_values.size < 2:
        return 1.0
    before_mean = float(before_values.mean())
    after_mean = float(after_values.mean())
    before_var = float(before_values.var())
    after_var = float(after_values.var())
    covariance = float(((before_values - before_mean) * (after_values - after_mean)).mean())
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * before_mean * after_mean + c1) * (2 * covariance + c2)
    denominator = (before_mean**2 + after_mean**2 + c1) * (before_var + after_var + c2)
    if denominator <= 0:
        return 1.0
    return float(np.clip(numerator / denominator, -1, 1))


def _ssim_metrics(before: dict[str, np.ndarray], after: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, float]:
    before_rgb = (before["B04"].astype(np.float32) + before["B03"] + before["B02"]) / 3.0
    after_rgb = (after["B04"].astype(np.float32) + after["B03"] + after["B02"]) / 3.0
    before_false_color = (before["B08"].astype(np.float32) + before["B04"] + before["B03"]) / 3.0
    after_false_color = (after["B08"].astype(np.float32) + after["B04"] + after["B03"]) / 3.0

    before_rgb, after_rgb = _robust_grayscale(before_rgb, after_rgb, valid)
    before_false_color, after_false_color = _robust_grayscale(before_false_color, after_false_color, valid)
    rgb_ssim = _masked_ssim(before_rgb, after_rgb, valid)
    false_color_ssim = _masked_ssim(before_false_color, after_false_color, valid)
    structural_change = 1.0 - ((rgb_ssim + false_color_ssim) / 2.0)
    return {
        "ssim_rgb": rgb_ssim,
        "ssim_false_color": false_color_ssim,
        "ssim_structural_change": structural_change,
    }


def _download_clay_checkpoint() -> None:
    CLAY_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=CLAY_CHECKPOINT_PATH.parent, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            with niquests.get(CLAY_CHECKPOINT_URL, stream=True, timeout=300) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                    if chunk:
                        temporary_file.write(chunk)
            temporary_file.flush()
            if temporary_path.stat().st_size < CLAY_MIN_CHECKPOINT_BYTES:
                raise ValueError(  # noqa: TRY301
                    f"Downloaded Clay checkpoint is unexpectedly small: {temporary_path.stat().st_size} bytes"
                )
            temporary_path.replace(CLAY_CHECKPOINT_PATH)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def _ensure_clay_checkpoint() -> Path:
    if CLAY_CHECKPOINT_PATH.exists() and CLAY_CHECKPOINT_PATH.stat().st_size >= CLAY_MIN_CHECKPOINT_BYTES:
        return CLAY_CHECKPOINT_PATH
    CLAY_CHECKPOINT_PATH.unlink(missing_ok=True)
    _download_clay_checkpoint()
    return CLAY_CHECKPOINT_PATH


@lru_cache
def _clay_metadata() -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    with CLAY_METADATA_PATH.open("r") as metadata_file:
        return yaml.safe_load(metadata_file)


@lru_cache
def _clay_model() -> Any:
    import timm  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from claymodel.module import ClayMAEModule  # noqa: PLC0415

    checkpoint_path = _ensure_clay_checkpoint()
    original_create_model = timm.create_model

    def create_model_without_pretrained_weights(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    try:
        timm.create_model = create_model_without_pretrained_weights
        model = ClayMAEModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            model_size="large",
            metadata_path=CLAY_METADATA_PATH.as_posix(),
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ratio=0.0,
            shuffle=False,
        )
    except Exception:  # noqa: BLE001
        CLAY_CHECKPOINT_PATH.unlink(missing_ok=True)
        _download_clay_checkpoint()
        model = ClayMAEModule.load_from_checkpoint(
            CLAY_CHECKPOINT_PATH,
            map_location="cpu",
            model_size="large",
            metadata_path=CLAY_METADATA_PATH.as_posix(),
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ratio=0.0,
            shuffle=False,
        )
    finally:
        timm.create_model = original_create_model

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return model.to(torch.device("cpu")).eval()


def _normalize_latlon(latitude: float, longitude: float) -> tuple[tuple[float, float], tuple[float, float]]:
    lat_radians = latitude * np.pi / 180
    lon_radians = longitude * np.pi / 180
    return (math.sin(lat_radians), math.cos(lat_radians)), (math.sin(lon_radians), math.cos(lon_radians))


def _normalize_timestamp(value: str | None) -> tuple[tuple[float, float], tuple[float, float]]:
    timestamp = datetime.fromisoformat(value) if value else datetime.utcnow()
    week = timestamp.isocalendar().week * 2 * np.pi / 52
    hour = timestamp.hour * 2 * np.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def _clay_band_metadata() -> tuple[list[str], list[float], list[float], list[float]]:
    sensor = _clay_metadata()[CLAY_PLATFORM]
    band_order = list(sensor["band_order"])
    means = [float(sensor["bands"]["mean"][band]) for band in band_order]
    stds = [float(sensor["bands"]["std"][band]) for band in band_order]
    wavelengths = [float(sensor["bands"]["wavelength"][band]) for band in band_order]
    return band_order, means, stds, wavelengths


def _clay_pixels(arrays: dict[str, np.ndarray]) -> Any:
    import torch  # noqa: PLC0415
    from torch.nn import functional  # noqa: PLC0415

    band_to_asset = {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "rededge1": "B05",
        "rededge2": "B06",
        "rededge3": "B07",
        "nir": "B08",
        "nir08": "B8A",
        "swir16": "B11",
        "swir22": "B12",
    }
    band_order, means, stds, _ = _clay_band_metadata()
    stack = np.stack([arrays[band_to_asset[band]].astype(np.float32) for band in band_order], axis=0)
    pixels = torch.from_numpy(np.ascontiguousarray(stack)).unsqueeze(0)
    if pixels.shape[-2:] != (CLAY_INPUT_SIZE, CLAY_INPUT_SIZE):
        pixels = functional.interpolate(
            pixels,
            size=(CLAY_INPUT_SIZE, CLAY_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    mean_tensor = torch.tensor(means, dtype=torch.float32).view(1, -1, 1, 1)
    std_tensor = torch.tensor(stds, dtype=torch.float32).view(1, -1, 1, 1)
    return (pixels - mean_tensor) / std_tensor


def _clay_patch_embeddings(
    arrays: dict[str, np.ndarray], latitude: float, longitude: float, acquisition_time: str | None
) -> np.ndarray:
    import torch  # noqa: PLC0415

    _, _, _, wavelengths = _clay_band_metadata()
    week_norm, hour_norm = _normalize_timestamp(acquisition_time)
    lat_norm, lon_norm = _normalize_latlon(latitude, longitude)
    model_input = {
        "platform": CLAY_PLATFORM,
        "time": torch.tensor(np.hstack((week_norm, hour_norm)).reshape(1, 4), dtype=torch.float32),
        "latlon": torch.tensor(np.hstack((lat_norm, lon_norm)).reshape(1, 4), dtype=torch.float32),
        "pixels": _clay_pixels(arrays),
        "gsd": torch.tensor([10.0], dtype=torch.float32),
        "waves": torch.tensor(wavelengths, dtype=torch.float32),
    }
    model = _clay_model()
    with torch.no_grad():
        encoded_patches, _, _, _ = model.model.encoder(model_input)
        patch_embeddings = encoded_patches[:, 1:, :]
        embeddings = patch_embeddings.detach().cpu().numpy()[0]
    return embeddings.astype(np.float32)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1, 1))


def _cosine_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominators = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominators > 0
    similarities = np.zeros(left.shape[0], dtype=np.float32)
    similarities[valid] = np.sum(left[valid] * right[valid], axis=1) / denominators[valid]
    return 1.0 - np.clip(similarities, -1, 1)


def _clay_change_metrics(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    site: Site,
    before_metadata: dict[str, Any],
    after_metadata: dict[str, Any],
) -> dict[str, float]:
    before, after, common_shape = _align_common_shape(before, after)
    before_patches = _clay_patch_embeddings(before, site.latitude, site.longitude, before_metadata.get("acquisition_time"))
    after_patches = _clay_patch_embeddings(after, site.latitude, site.longitude, after_metadata.get("acquisition_time"))
    patch_distances = _cosine_distances(before_patches, after_patches)
    grid_size = int(math.sqrt(patch_distances.size))
    patch_distance_map = patch_distances.reshape(grid_size, grid_size)
    patch_threshold = _mad_threshold(patch_distances, minimum=0.05)
    patch_changed = patch_distance_map > patch_threshold
    crop_area_m2 = common_shape[0] * common_shape[1] * _pixel_area_m2(before_metadata)
    clay_component_metrics = _connected_component_metrics(patch_changed, crop_area_m2 / patch_distances.size)
    mean_before_embedding = before_patches.mean(axis=0)
    mean_after_embedding = after_patches.mean(axis=0)
    mean_similarity = _cosine_similarity(mean_before_embedding, mean_after_embedding)
    mean_distance = 1.0 - mean_similarity
    top_decile_threshold = np.nanpercentile(patch_distances, 90)
    top_decile = patch_distances[patch_distances >= top_decile_threshold]
    return {
        "clay_cosine_similarity": mean_similarity,
        "clay_cosine_distance": mean_distance,
        "clay_patch_distance_median": float(np.nanmedian(patch_distances)),
        "clay_patch_distance_p90": _safe_percentile(patch_distances, 90),
        "clay_patch_distance_p95": _safe_percentile(patch_distances, 95),
        "clay_patch_distance_top_decile_mean": float(np.nanmean(top_decile)) if top_decile.size else 0.0,
        "clay_patch_distance_threshold": float(patch_threshold),
        "clay_patch_changed_fraction": float(patch_changed.mean()),
        "clay_patch_largest_component_area_ha": clay_component_metrics["largest_component_area_ha"],
        "clay_patch_changed_area_ha": clay_component_metrics["changed_area_ha"],
        "clay_patch_component_count": clay_component_metrics["component_count"],
        "clay_embedding_dim": float(before_patches.shape[1]),
        "clay_patch_count": float(patch_distances.size),
    }


def _compute_change(  # noqa: PLR0915
    site: Site,
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    before_metadata: dict[str, Any],
    clay_metrics: dict[str, float],
) -> dict[str, Any]:
    before, after, common_shape = _align_common_shape(before, after)
    before_indices = _indices(before)
    after_indices = _indices(after)
    valid = ~(np.isin(before["SCL"], list(INVALID_SCL_CLASSES | BAD_CLOUD_SCL_CLASSES)))
    valid &= ~(np.isin(after["SCL"], list(INVALID_SCL_CLASSES | BAD_CLOUD_SCL_CLASSES)))
    valid &= before["B04"] > 0
    valid &= after["B04"] > 0

    if int(valid.sum()) == 0:
        return {
            "site_id": site.site_id,
            "name": site.name,
            "latitude": site.latitude,
            "longitude": site.longitude,
            "status": "no_valid_pixels",
            "score": 0.0,
        }

    delta_ndbi_map = after_indices["ndbi"] - before_indices["ndbi"]
    delta_bsi_map = after_indices["bsi"] - before_indices["bsi"]
    delta_ndvi_loss_map = before_indices["ndvi"] - after_indices["ndvi"]
    delta_brightness_map = (after_indices["brightness"] - before_indices["brightness"]) / 10_000.0
    delta_ndbi = delta_ndbi_map[valid]
    delta_bsi = delta_bsi_map[valid]
    delta_ndvi_loss = delta_ndvi_loss_map[valid]
    delta_brightness = delta_brightness_map[valid]
    after_mndwi = after_indices["mndwi"][valid]

    before_stack = _spectral_stack(before)
    after_stack = _spectral_stack(after)
    cva_map = np.sqrt(np.nanmean((after_stack - before_stack) ** 2, axis=0))
    cva_values = cva_map[valid]
    cva_threshold = _mad_threshold(cva_values, minimum=0.035)
    cva_changed = (cva_map > cva_threshold) & valid

    index_changed = (delta_ndbi_map > 0.12) | (delta_bsi_map > 0.10) | (delta_ndvi_loss_map > 0.15)
    brightness_changed = delta_brightness_map > 0.04
    changed_mask = cva_changed & (index_changed | brightness_changed)
    if int(changed_mask.sum()) == 0:
        changed_mask = cva_changed

    construction_spectral_mask = valid & (
        (
            (delta_ndbi_map > 0.08)
            & (after_indices["ndbi"] > 0.0)
            & (after_indices["ndvi"] < 0.45)
            & (after_indices["mndwi"] < 0.10)
        )
        | (
            (delta_bsi_map > 0.08)
            & (delta_ndvi_loss_map > 0.08)
            & (after_indices["bsi"] > 0.02)
            & (after_indices["ndvi"] < 0.40)
            & (after_indices["mndwi"] < 0.10)
        )
    )
    construction_mask = construction_spectral_mask & changed_mask

    pixel_area_m2 = _pixel_area_m2(before_metadata)
    component_metrics = _connected_component_metrics(changed_mask, pixel_area_m2)
    construction_component_metrics = _connected_component_metrics(construction_mask, pixel_area_m2)
    ssim = _ssim_metrics(before, after, valid)
    changed_pixel_fraction = float(changed_mask[valid].mean())
    construction_pixel_fraction = float(construction_mask[valid].mean())

    built_up_gain = min(_component_score(delta_ndbi, 0.02, 0.18), 25.0)
    bare_soil_gain = min(_component_score(delta_bsi, 0.02, 0.16), 25.0)
    vegetation_loss = min(_component_score(delta_ndvi_loss, 0.04, 0.25), 25.0)
    brightness_gain = min(_component_score(delta_brightness, 0.01, 0.18), 25.0)
    coherent_cva_component_area = _score_scalar(component_metrics["largest_component_area_ha"], 0.03, 3.0)
    coherent_cva_changed_fraction = _score_scalar(changed_pixel_fraction, 0.0001, 0.005)
    coherent_change_evidence = max(coherent_cva_component_area, coherent_cva_changed_fraction)
    construction_component_area = _score_scalar(construction_component_metrics["largest_component_area_ha"], 0.50, 30.0)
    construction_changed_area = _score_scalar(construction_component_metrics["changed_area_ha"], 2.0, 100.0)
    construction_fraction = _score_scalar(construction_pixel_fraction, 0.005, 0.10)
    construction_evidence = max(construction_component_area, construction_changed_area, construction_fraction)
    clay_patch_intensity = max(
        _score_scalar(clay_metrics["clay_patch_distance_p95"], 0.16, 0.45),
        _score_scalar(clay_metrics["clay_patch_distance_top_decile_mean"], 0.18, 0.50),
    )
    clay_patch_area = max(
        _score_scalar(clay_metrics["clay_patch_largest_component_area_ha"], 5.0, 50.0),
        _score_scalar(clay_metrics["clay_patch_changed_fraction"], 0.05, 0.20),
    )
    clay_patch_cluster = math.sqrt(clay_patch_intensity * clay_patch_area)
    structural_change = _score_scalar(ssim["ssim_structural_change"], 0.03, 0.35)
    clay_embedding_change = _score_scalar(clay_metrics["clay_cosine_distance"], 0.02, 0.25)
    object_evidence = max(construction_evidence, clay_patch_cluster)
    global_change_gate = _score_scalar(object_evidence, 20.0, 70.0) / 100.0
    water_penalty = float(np.clip((np.nanmean(after_mndwi > 0.2) - 0.1) / 0.4, 0, 1) * 20)
    weak_construction_direction_penalty = 0.0
    if (
        clay_patch_intensity < 50.0
        and built_up_gain <= 1.0
        and bare_soil_gain <= 1.0
        and brightness_gain <= 1.0
        and structural_change < 70.0
    ):
        weak_construction_direction_penalty = (50.0 - clay_patch_intensity) * 2.0
    coherent_change_cap = 25.0 + 0.75 * coherent_change_evidence
    raw_score = (
        0.35 * clay_patch_cluster
        + 0.30 * construction_evidence
        + 0.15 * coherent_cva_component_area * global_change_gate
        + 0.05 * structural_change * global_change_gate
        + 0.05 * built_up_gain
        + 0.04 * bare_soil_gain
        + 0.03 * vegetation_loss
        + 0.03 * brightness_gain
    )
    score = max(
        0.0,
        min(raw_score, coherent_change_cap) - water_penalty - weak_construction_direction_penalty,
    )

    return {
        "site_id": site.site_id,
        "name": site.name,
        "latitude": site.latitude,
        "longitude": site.longitude,
        "operators": site.operators,
        "source_count": site.source_count,
        "source_ids": site.source_ids,
        "status": "scored",
        "score": round(float(score), 4),
        "component_scores": {
            "built_up_gain": round(built_up_gain, 4),
            "bare_soil_or_construction_gain": round(bare_soil_gain, 4),
            "vegetation_loss": round(vegetation_loss, 4),
            "brightness_gain": round(brightness_gain, 4),
            "coherent_cva_component_area": round(coherent_cva_component_area, 4),
            "coherent_cva_changed_fraction": round(coherent_cva_changed_fraction, 4),
            "coherent_change_evidence": round(coherent_change_evidence, 4),
            "construction_component_area": round(construction_component_area, 4),
            "construction_changed_area": round(construction_changed_area, 4),
            "construction_fraction": round(construction_fraction, 4),
            "construction_evidence": round(construction_evidence, 4),
            "ssim_structural_change": round(structural_change, 4),
            "clay_patch_intensity": round(clay_patch_intensity, 4),
            "clay_patch_area": round(clay_patch_area, 4),
            "clay_patch_cluster": round(clay_patch_cluster, 4),
            "clay_embedding_change": round(clay_embedding_change, 4),
            "object_evidence": round(object_evidence, 4),
            "global_change_gate": round(global_change_gate, 4),
            "coherent_change_cap": round(coherent_change_cap, 4),
            "weak_construction_direction_penalty": round(weak_construction_direction_penalty, 4),
            "water_penalty": round(water_penalty, 4),
        },
        "metrics": {
            "clay_cosine_similarity": round(clay_metrics["clay_cosine_similarity"], 6),
            "clay_cosine_distance": round(clay_metrics["clay_cosine_distance"], 6),
            "clay_patch_distance_median": round(clay_metrics["clay_patch_distance_median"], 6),
            "clay_patch_distance_p90": round(clay_metrics["clay_patch_distance_p90"], 6),
            "clay_patch_distance_p95": round(clay_metrics["clay_patch_distance_p95"], 6),
            "clay_patch_distance_top_decile_mean": round(clay_metrics["clay_patch_distance_top_decile_mean"], 6),
            "clay_patch_distance_threshold": round(clay_metrics["clay_patch_distance_threshold"], 6),
            "clay_patch_changed_fraction": round(clay_metrics["clay_patch_changed_fraction"], 6),
            "clay_patch_largest_component_area_ha": round(clay_metrics["clay_patch_largest_component_area_ha"], 4),
            "clay_patch_changed_area_ha": round(clay_metrics["clay_patch_changed_area_ha"], 4),
            "clay_patch_component_count": int(clay_metrics["clay_patch_component_count"]),
            "clay_embedding_dim": int(clay_metrics["clay_embedding_dim"]),
            "clay_patch_count": int(clay_metrics["clay_patch_count"]),
            "valid_pixel_fraction": round(float(valid.sum() / valid.size), 6),
            "changed_pixel_fraction": round(changed_pixel_fraction, 6),
            "construction_pixel_fraction": round(construction_pixel_fraction, 6),
            "construction_spectral_pixel_fraction": round(float(construction_spectral_mask[valid].mean()), 6),
            "cva_threshold": round(float(cva_threshold), 6),
            "cva_median": round(float(np.nanmedian(cva_values)), 6),
            "cva_p90": round(_safe_percentile(cva_values, 90), 6),
            "cva_p95": round(_safe_percentile(cva_values, 95), 6),
            "changed_area_ha": round(component_metrics["changed_area_ha"], 4),
            "largest_component_area_ha": round(component_metrics["largest_component_area_ha"], 4),
            "largest_component_fraction": round(component_metrics["largest_component_fraction"], 6),
            "component_count": int(component_metrics["component_count"]),
            "construction_changed_area_ha": round(construction_component_metrics["changed_area_ha"], 4),
            "construction_largest_component_area_ha": round(
                construction_component_metrics["largest_component_area_ha"], 4
            ),
            "construction_largest_component_fraction": round(
                construction_component_metrics["largest_component_fraction"], 6
            ),
            "construction_component_count": int(construction_component_metrics["component_count"]),
            "ssim_rgb": round(ssim["ssim_rgb"], 6),
            "ssim_false_color": round(ssim["ssim_false_color"], 6),
            "ssim_structural_change": round(ssim["ssim_structural_change"], 6),
            "common_crop_height": common_shape[0],
            "common_crop_width": common_shape[1],
            "delta_ndbi_median": round(float(np.nanmedian(delta_ndbi)), 6),
            "delta_bsi_median": round(float(np.nanmedian(delta_bsi)), 6),
            "delta_ndvi_loss_median": round(float(np.nanmedian(delta_ndvi_loss)), 6),
            "delta_brightness_median": round(float(np.nanmedian(delta_brightness)), 6),
        },
    }


class RankDataCenterBuildout(Task):
    csv_url: str = DEFAULT_SITES_CSV_URL
    max_sites: int | None = None
    random_seed: int = 1337
    before_date: str = "2024-05-01"
    after_date: str = "2026-05-01"
    window_days: int = 60
    crop_size_m: int = 3000
    scene_cloud_cover_max: float = 30.0
    crop_cloud_cover_max: float = 1.0
    status_filter: list[str] | None = None

    @staticmethod
    def identifier() -> tuple[str, str]:
        return "tilebox.com/datacenters/RankDataCenterBuildout", "v1.12"

    def execute(self, context: ExecutionContext):  # noqa: ANN201
        context.current_task.display = "RankDataCenterBuildout"
        status_filter = self.status_filter if self.status_filter is not None else DEFAULT_STATUS_FILTER
        sites = _merge_sites(self.csv_url, self.max_sites, self.random_seed, status_filter)
        context.job_cache["sites.json"] = _json_dumps([asdict(site) for site in sites])
        context.logger.info(
            "Loaded, merged, and sampled sites",
            input_url=self.csv_url,
            site_count=len(sites),
            random_seed=self.random_seed,
            status_filter=", ".join(status_filter),
        )

        scene_tasks = []
        for site in sites:
            scene_tasks.extend(
                [
                    SelectAndCacheScene(
                        site_id=site.site_id,
                        label="before",
                        target_date=self.before_date,
                        window_days=self.window_days,
                        crop_size_m=self.crop_size_m,
                        scene_cloud_cover_max=self.scene_cloud_cover_max,
                        crop_cloud_cover_max=self.crop_cloud_cover_max,
                    ),
                    SelectAndCacheScene(
                        site_id=site.site_id,
                        label="after",
                        target_date=self.after_date,
                        window_days=self.window_days,
                        crop_size_m=self.crop_size_m,
                        scene_cloud_cover_max=self.scene_cloud_cover_max,
                        crop_cloud_cover_max=self.crop_cloud_cover_max,
                    ),
                ]
            )
            context.progress("scenes").add(2)

        context.logger.info("Submitting scene selection stage", scene_task_count=len(scene_tasks))
        scene_handles = context.submit_subtasks(scene_tasks, max_retries=2)
        context.logger.info("Submitting site change compute stage", site_count=len(sites))
        compute_handles = context.submit_subtasks(
            [ComputeSiteChange(site_id=site.site_id) for site in sites],
            depends_on=scene_handles,
        )
        context.submit_subtask(WriteRankingOutput(), depends_on=compute_handles)


class SelectAndCacheScene(Task):
    site_id: str
    label: str
    target_date: str
    window_days: int = 30
    crop_size_m: int = 3000
    scene_cloud_cover_max: float = 30.0
    crop_cloud_cover_max: float = 1.0

    @staticmethod
    def identifier() -> tuple[str, str]:
        return "tilebox.com/datacenters/SelectAndCacheScene", "v1.12"

    def execute(self, context: ExecutionContext):  # noqa: ANN201, PLR0915
        site = _sites_by_id(context.job_cache["sites.json"])[self.site_id]
        context.current_task.display = f"Select {self.label} {site.site_id}"
        metadata_key = f"scenes/{site.site_id}/{self.label}/metadata.json"
        bands_key = f"scenes/{site.site_id}/{self.label}/bands.npz"
        preview_key = f"scenes/{site.site_id}/{self.label}/preview.png"
        log = context.logger.bind(site_id=site.site_id, label=self.label, target_date=self.target_date)

        try:
            candidates = _dataset_candidates(
                site.latitude,
                site.longitude,
                self.target_date,
                self.window_days,
                self.crop_size_m,
                self.scene_cloud_cover_max,
            )
            candidate_names = [candidate["granule_name"] for candidate in candidates]
            log.info(
                "Queried Sentinel-2 candidates",
                candidate_count=len(candidates),
                candidate_granule_names=", ".join(candidate_names),
            )
            if not candidates:
                log.info("No Sentinel-2 candidates found", candidate_granule_names="")
                metadata = SceneMetadata(
                    status="no_candidate_scene",
                    site_id=site.site_id,
                    label=self.label,
                    message="Tilebox query returned no low-cloud Sentinel-2 L2A candidates",
                )
                context.job_cache[metadata_key] = _json_dumps(asdict(metadata))
                context.progress("scenes").done(1)
                return

            skipped_granule_names = []
            for candidate in candidates:
                with context.tracer.span("list-copernicus-assets") as span:
                    span.set_attribute("scene_id", candidate["granule_name"])
                    span.set_attribute("data_location", candidate["location"])
                    assets = _find_copernicus_jp2_assets(candidate["location"])
                    missing_assets = sorted(set(JP2_BAND_ASSET_SUFFIXES) - set(assets))
                    span.set_attribute("asset_count", len(assets))
                    span.set_attribute("asset_format", "jp2")
                    span.set_attribute("missing_assets", ",".join(missing_assets))

                if missing_assets:
                    skipped_granule_names.append(candidate["granule_name"])
                    log.info(
                        "Skipped candidate because expected Copernicus JP2 assets were not found",
                        skip_reason="missing_copernicus_jp2_assets",
                        scene_id=candidate["granule_name"],
                        data_location=candidate["location"],
                        found_asset_names=", ".join(sorted(assets)),
                        missing_assets=", ".join(missing_assets),
                        scene_cloud_cover=candidate["cloud_cover"],
                    )
                    continue

                with context.tracer.span("download-cropped-assets") as span:
                    span.set_attribute("scene_id", candidate["granule_name"])
                    span.set_attribute("data_location", candidate["location"])
                    span.set_attribute("asset_format", "jp2")
                    for band_name, asset_path in assets.items():
                        span.set_attribute(f"asset.{band_name}", asset_path)
                    try:
                        arrays, crop_metadata = _read_crop(
                            assets,
                            site.latitude,
                            site.longitude,
                            self.crop_size_m,
                        )
                        span.set_attribute("crop_height", crop_metadata["height"])
                        span.set_attribute("crop_width", crop_metadata["width"])
                    except Exception as error:  # noqa: BLE001
                        span.set_attribute("error", str(error))
                        skipped_granule_names.append(candidate["granule_name"])
                        log.info(
                            "Skipped candidate because Copernicus crop read failed",
                            skip_reason="copernicus_asset_read_failed",
                            scene_id=candidate["granule_name"],
                            data_location=candidate["location"],
                            asset_format="jp2",
                            error=str(error),
                            scene_cloud_cover=candidate["cloud_cover"],
                        )
                        continue

                expected_crop_pixels = _expected_crop_pixels(self.crop_size_m)
                if not _has_full_crop_size(crop_metadata, self.crop_size_m):
                    skipped_granule_names.append(candidate["granule_name"])
                    log.info(
                        "Skipped candidate because crop did not cover the full target area",
                        skip_reason="partial_crop_overlap",
                        scene_id=candidate["granule_name"],
                        data_location=candidate["location"],
                        crop_height=crop_metadata["height"],
                        crop_width=crop_metadata["width"],
                        expected_min_crop_pixels=expected_crop_pixels,
                        scene_cloud_cover=candidate["cloud_cover"],
                    )
                    continue

                crop_invalid_percent = _invalid_data_fraction(arrays) * 100
                if crop_invalid_percent > MAX_CROP_INVALID_PERCENT:
                    skipped_granule_names.append(candidate["granule_name"])
                    log.info(
                        "Skipped candidate because crop contains too much invalid or padded data",
                        skip_reason="crop_invalid_data_too_high",
                        scene_id=candidate["granule_name"],
                        data_location=candidate["location"],
                        crop_invalid_percent=crop_invalid_percent,
                        crop_invalid_percent_max=MAX_CROP_INVALID_PERCENT,
                        scene_cloud_cover=candidate["cloud_cover"],
                    )
                    continue

                crop_cloud_cover = _bad_fraction(arrays["SCL"]) * 100
                log.info(
                    "Computed crop cloud cover",
                    scene_id=candidate["granule_name"],
                    data_location=candidate["location"],
                    crop_cloud_cover=crop_cloud_cover,
                    crop_invalid_percent=crop_invalid_percent,
                    scene_cloud_cover=candidate["cloud_cover"],
                )
                if crop_cloud_cover > self.crop_cloud_cover_max:
                    skipped_granule_names.append(candidate["granule_name"])
                    log.info(
                        "Skipped candidate because crop cloud cover was too high",
                        skip_reason="crop_cloud_cover_too_high",
                        scene_id=candidate["granule_name"],
                        data_location=candidate["location"],
                        crop_cloud_cover=crop_cloud_cover,
                        crop_cloud_cover_max=self.crop_cloud_cover_max,
                        scene_cloud_cover=candidate["cloud_cover"],
                    )
                    continue

                crop_metadata.update(
                    {
                        "data_location": candidate["location"],
                        "asset_format": "jp2",
                        "asset_paths": assets,
                        "scene_id": candidate["granule_name"],
                        "acquisition_time": candidate["time"].isoformat(),
                        "crop_invalid_percent": crop_invalid_percent,
                    }
                )
                with context.tracer.span("cache-cropped-assets") as span:
                    bands_bytes = _save_npz(arrays, crop_metadata)
                    preview_bytes = _preview_png(arrays)
                    span.set_attribute("bands_key", bands_key)
                    span.set_attribute("bands_bytes", len(bands_bytes))
                    span.set_attribute("preview_key", preview_key)
                    span.set_attribute("preview_bytes", len(preview_bytes))
                    context.job_cache[bands_key] = bands_bytes
                    context.job_cache[preview_key] = preview_bytes
                context.progress("scenes").done(1)
                metadata = SceneMetadata(
                    status="selected",
                    site_id=site.site_id,
                    label=self.label,
                    scene_id=candidate["granule_name"],
                    acquisition_time=candidate["time"].isoformat(),
                    crop_cloud_cover=crop_cloud_cover,
                    crop_invalid_percent=crop_invalid_percent,
                    scene_cloud_cover=candidate["cloud_cover"],
                    bands_key=bands_key,
                    preview_key=preview_key,
                    data_location=candidate["location"],
                    asset_format="jp2",
                )
                context.job_cache[metadata_key] = _json_dumps(asdict(metadata))
                return

            log.info(
                "No suitable scene found",
                candidate_count=len(candidates),
                candidate_granule_names=", ".join(candidate_names),
                skipped_granule_names=", ".join(skipped_granule_names),
                skipped_count=len(skipped_granule_names),
            )
            metadata = SceneMetadata(
                status="no_clear_scene",
                site_id=site.site_id,
                label=self.label,
                message="No candidate met the target crop cloud threshold",
            )
            context.job_cache[metadata_key] = _json_dumps(asdict(metadata))
            context.progress("scenes").done(1)
        except Exception:
            log.exception("Scene selection failed")
            context.progress("scenes").done(1)
            raise


class ComputeSiteChange(Task):
    site_id: str

    @staticmethod
    def identifier() -> tuple[str, str]:
        return "tilebox.com/datacenters/ComputeSiteChange", "v1.12"

    def execute(self, context: ExecutionContext):  # noqa: ANN201
        site = _sites_by_id(context.job_cache["sites.json"])[self.site_id]
        context.current_task.display = f"Compute {site.site_id}"
        before_metadata = _json_loads(context.job_cache[f"scenes/{site.site_id}/before/metadata.json"])
        after_metadata = _json_loads(context.job_cache[f"scenes/{site.site_id}/after/metadata.json"])

        result: dict[str, Any]
        if before_metadata["status"] != "selected" or after_metadata["status"] != "selected":
            result = {
                "site_id": site.site_id,
                "name": site.name,
                "latitude": site.latitude,
                "longitude": site.longitude,
                "operators": site.operators,
                "source_count": site.source_count,
                "source_ids": site.source_ids,
                "status": "missing_scene_pair",
                "score": 0.0,
                "before_scene": before_metadata,
                "after_scene": after_metadata,
            }
        else:
            before_arrays, before_crop_metadata = _load_npz(context.job_cache[before_metadata["bands_key"]])
            after_arrays, _ = _load_npz(context.job_cache[after_metadata["bands_key"]])
            with context.tracer.span("clay-inference") as span:
                span.set_attribute("site_id", site.site_id)
                span.set_attribute("before_scene_id", before_metadata.get("scene_id") or "")
                span.set_attribute("after_scene_id", after_metadata.get("scene_id") or "")
                clay_metrics = _clay_change_metrics(
                    before_arrays,
                    after_arrays,
                    site,
                    before_metadata,
                    after_metadata,
                )
                span.set_attribute("clay_cosine_similarity", clay_metrics["clay_cosine_similarity"])
                span.set_attribute("clay_cosine_distance", clay_metrics["clay_cosine_distance"])
                span.set_attribute("clay_patch_distance_p95", clay_metrics["clay_patch_distance_p95"])
                span.set_attribute("clay_patch_changed_fraction", clay_metrics["clay_patch_changed_fraction"])
                span.set_attribute(
                    "clay_patch_largest_component_area_ha",
                    clay_metrics["clay_patch_largest_component_area_ha"],
                )
            result = _compute_change(site, before_arrays, after_arrays, before_crop_metadata, clay_metrics)
            result["before_scene"] = before_metadata
            result["after_scene"] = after_metadata

        context.job_cache[f"results/{site.site_id}.json"] = _json_dumps(result)


class WriteRankingOutput(Task):

    @staticmethod
    def identifier() -> tuple[str, str]:
        return "tilebox.com/datacenters/WriteRankingOutput", "v1.12"

    def execute(self, context: ExecutionContext):  # noqa: ANN201
        site_ids = list(_sites_by_id(context.job_cache["sites.json"]))
        context.current_task.display = f"WriteRankingOutput(n={len(site_ids)})"
        results = [_json_loads(context.job_cache[f"results/{site_id}.json"]) for site_id in site_ids]
        results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        output = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "ranking": results,
        }
        context.job_cache["outputs/ranking.json"] = _json_dumps(output)


runner = Runner(
    tasks=[RankDataCenterBuildout, SelectAndCacheScene, ComputeSiteChange, WriteRankingOutput],
    cache=workflow_cache(),
)
