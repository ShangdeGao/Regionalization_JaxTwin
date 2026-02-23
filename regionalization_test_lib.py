"""
regionalization_test_lib.py
Library for testing regionalization schemes using the neighbor-contrast method.
Import this in a Colab notebook and call the high-level functions.
"""

import os
import tempfile
import zipfile
import glob as globmod
import warnings
import urllib.request

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, gaussian_kde
from shapely.geometry import Point
from IPython.display import display

warnings.filterwarnings("ignore")

COMMON_CRS = "EPSG:4326"

# ── Default region sources & ID columns ──

REPO_BASE = "https://github.com/ShangdeGao/Regionalization_JaxTwin/raw/main"

DEFAULT_REGION_SOURCES = {
    "Regionalization Output":    f"{REPO_BASE}/Regionalization%20Output.zip",
    "TAZ":           f"{REPO_BASE}/TAZ_Test.zip",
    "Census Tracts": f"{REPO_BASE}/Census%20Tract_Test.zip",
    "Neighborhoods": f"{REPO_BASE}/Neighborhood_Test.zip",
}

DEFAULT_REGION_ID_COLS = {
    "Regionalization Output":    "value_gsd",
    "TAZ":           "TAZCE10",
    "Census Tracts": "GEOID",
    "Neighborhoods": "REGIONID",
}


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Utility functions
# ═══════════════════════════════════════════════════════════════════════

def ensure_crs_match(gdf, target_crs=COMMON_CRS):
    """Reproject GeoDataFrame to target CRS if needed."""
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs.to_epsg() != int(target_crs.split(":")[1]):
        gdf = gdf.to_crs(target_crs)
    return gdf


def detect_format(filename):
    """Detect file format from extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return "csv"
    elif ext in (".shp", ".geojson", ".json", ".gpkg"):
        return "vector"
    elif ext in (".tif", ".tiff"):
        return "raster"
    elif ext == ".zip":
        return "zip"
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def extract_zip(zip_path):
    """Extract a zip and return the path to the .shp (or geojson/gpkg) inside."""
    extract_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    shp_files = globmod.glob(os.path.join(extract_dir, "**", "*.shp"), recursive=True)
    if shp_files:
        return shp_files[0]
    for ext in ("*.geojson", "*.json", "*.gpkg"):
        found = globmod.glob(os.path.join(extract_dir, "**", ext), recursive=True)
        if found:
            return found[0]
    raise FileNotFoundError(f"No .shp, .geojson, or .gpkg found inside {zip_path}")


def remove_outliers_iqr(arr, factor=1.5):
    """IQR-based outlier removal on a 1D numpy array."""
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    mask = (arr >= q1 - factor * iqr) & (arr <= q3 + factor * iqr)
    return arr[mask]


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Load reference regions
# ═══════════════════════════════════════════════════════════════════════

def load_regions(sources=None, id_cols=None):
    """Download and load all region shapefiles.

    Returns:
        regions: dict {name: GeoDataFrame}
        region_id_cols: dict {name: column_name}
    """
    if sources is None:
        sources = DEFAULT_REGION_SOURCES
    if id_cols is None:
        id_cols = DEFAULT_REGION_ID_COLS

    regions = {}
    for name, url in sources.items():
        try:
            tmp_zip = os.path.join(tempfile.gettempdir(), f"{name.replace(' ', '_')}.zip")
            urllib.request.urlretrieve(url, tmp_zip)
            shp_path = extract_zip(tmp_zip)
            gdf = gpd.read_file(shp_path)
            gdf = ensure_crs_match(gdf)
            regions[name] = gdf
            print(f"  Loaded {name}: {len(gdf)} polygons, "
                  f"'{id_cols[name]}' has {gdf[id_cols[name]].nunique()} unique values")
        except Exception as e:
            print(f"  FAILED to load {name}: {e}")

    print(f"\nSuccessfully loaded {len(regions)}/{len(sources)} region sets.")
    return regions, id_cols


def preview_regions(regions, region_sources=None):
    """Show a 2x2 map of all region schemes."""
    if region_sources is None:
        region_sources = DEFAULT_REGION_SOURCES
    names = list(region_sources.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        if name in regions:
            regions[name].plot(ax=ax, edgecolor="black", linewidth=0.5,
                               color="lightblue", alpha=0.6)
            ax.set_title(name, fontsize=13, fontweight="bold")
        else:
            ax.text(0.5, 0.5, f"{name}\n(not loaded)",
                    ha="center", va="center", fontsize=12, color="red",
                    transform=ax.transAxes)
            ax.set_title(name, fontsize=13, color="red")
        ax.set_axis_off()
    plt.suptitle("Reference Regionalization Schemes", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Parse uploaded file
# ═══════════════════════════════════════════════════════════════════════

def parse_upload(uploaded_filename):
    """Parse an uploaded file. Returns (test_gdf_or_df, raster_path, file_format).

    - CSV: returns (DataFrame, None, 'csv')
    - Vector/zip: returns (GeoDataFrame, None, 'vector')
    - Raster: returns (None, raster_path, 'raster')
    """
    file_format = detect_format(uploaded_filename)
    test_data = None
    raster_path = None

    if file_format == "zip":
        shp_path = extract_zip(uploaded_filename)
        print(f"Extracted shapefile: {shp_path}")
        test_data = gpd.read_file(shp_path)
        test_data = ensure_crs_match(test_data)
        file_format = "vector"
        print(f"Vector loaded: {len(test_data)} features, columns: {list(test_data.columns)}")

    elif file_format == "csv":
        test_data = pd.read_csv(uploaded_filename)
        print(f"CSV loaded: {len(test_data)} rows, columns: {list(test_data.columns)}")

    elif file_format == "vector":
        test_data = gpd.read_file(uploaded_filename)
        test_data = ensure_crs_match(test_data)
        print(f"Vector loaded: {len(test_data)} features, columns: {list(test_data.columns)}")

    elif file_format == "raster":
        import rasterio
        raster_path = uploaded_filename
        with rasterio.open(raster_path) as src:
            print(f"Raster loaded: {src.width}x{src.height}, {src.count} band(s), CRS={src.crs}")

    return test_data, raster_path, file_format


def build_test_gdf(test_data, file_format, lat_col=None, lon_col=None, value_col=None):
    """Build a standardized GeoDataFrame with a 'test_value' column.

    For CSV: provide lat_col, lon_col, value_col.
    For vector: provide value_col (the numeric attribute).
    For raster: returns None (zonal stats computed later).
    """
    if file_format == "csv":
        geometry = [Point(xy) for xy in zip(test_data[lon_col], test_data[lat_col])]
        gdf = gpd.GeoDataFrame(test_data, geometry=geometry, crs=COMMON_CRS)
        gdf["test_value"] = gdf[value_col]
        print(f"Created point GeoDataFrame: {len(gdf)} points, value column = '{value_col}'")
        return gdf

    elif file_format == "vector":
        test_data["test_value"] = test_data[value_col]
        print(f"Using attribute '{value_col}' as test variable ({len(test_data)} features)")
        return test_data

    else:
        return None


def upload_and_select():
    """Upload a file in Colab, parse it, show column-selection widgets, and return
    a dict that the user passes to ``confirm_selection()`` after choosing columns.

    Returns:
        context: dict with keys test_data, raster_path, file_format, and widget refs
    """
    import ipywidgets as widgets

    from google.colab import files as colab_files
    uploaded = colab_files.upload()
    uploaded_filename = list(uploaded.keys())[0]

    test_data, raster_path, file_format = parse_upload(uploaded_filename)

    ctx = {
        "test_data": test_data,
        "raster_path": raster_path,
        "file_format": file_format,
    }

    if file_format == "csv":
        cols = list(test_data.columns)
        numeric_cols = list(test_data.select_dtypes(include=[np.number]).columns)
        w_lat = widgets.Dropdown(
            options=cols, description="Latitude:",
            value=next((c for c in cols if c.lower() in ("lat", "latitude", "y")), cols[0]))
        w_lon = widgets.Dropdown(
            options=cols, description="Longitude:",
            value=next((c for c in cols if c.lower() in ("lon", "lng", "longitude", "x")), cols[0]))
        w_val = widgets.Dropdown(
            options=numeric_cols, description="Value col:",
            value=numeric_cols[0] if numeric_cols else cols[0])
        display(w_lat, w_lon, w_val)
        ctx["w_lat"] = w_lat
        ctx["w_lon"] = w_lon
        ctx["w_val"] = w_val

    elif file_format == "vector":
        numeric_cols = list(test_data.select_dtypes(include=[np.number]).columns)
        w_val = widgets.Dropdown(
            options=numeric_cols, description="Attribute:",
            value=numeric_cols[0] if numeric_cols else None)
        display(w_val)
        ctx["w_val"] = w_val

    elif file_format == "raster":
        import rasterio
        with rasterio.open(raster_path) as src:
            n_bands = src.count
        w_band = widgets.Dropdown(
            options=[str(i) for i in range(1, n_bands + 1)],
            description="Band:", value="1")
        display(w_band)
        ctx["w_band"] = w_band

    return ctx


def confirm_selection(ctx):
    """Read widget values from *ctx* (returned by ``upload_and_select``) and
    build the test GeoDataFrame.

    Returns:
        test_gdf: GeoDataFrame (or None for raster)
        raster_path: str (or None for vector/csv)
        raster_band: int
    """
    file_format = ctx["file_format"]
    test_data = ctx["test_data"]
    raster_path = ctx["raster_path"]
    test_gdf = None
    raster_band = 1

    if file_format == "csv":
        test_gdf = build_test_gdf(
            test_data, file_format,
            lat_col=ctx["w_lat"].value,
            lon_col=ctx["w_lon"].value,
            value_col=ctx["w_val"].value)
    elif file_format == "vector":
        test_gdf = build_test_gdf(
            test_data, file_format,
            value_col=ctx["w_val"].value)
    elif file_format == "raster":
        raster_band = int(ctx["w_band"].value)
        print(f"Using raster band {raster_band}")

    return test_gdf, raster_path, raster_band


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Neighbor-contrast analysis
# ═══════════════════════════════════════════════════════════════════════

def _build_polygon_adjacency(region_gdf, region_id_col):
    """Build adjacency dict: {region_id: set of neighbor region_ids}."""
    from shapely.strtree import STRtree

    gdf = region_gdf.copy().reset_index(drop=True)
    ids = gdf[region_id_col].values
    geoms = gdf.geometry.values

    tree = STRtree(geoms)
    adjacency = {rid: set() for rid in ids}

    for i, geom in enumerate(geoms):
        candidates = tree.query(geom)
        for j in candidates:
            if i != j and (geoms[i].touches(geoms[j]) or
                           (geoms[i].intersects(geoms[j]) and not geoms[i].within(geoms[j]))):
                adjacency[ids[i]].add(ids[j])
                adjacency[ids[j]].add(ids[i])

    return adjacency


def _spatial_join_points(test_gdf, region_gdf, region_id_col, value_col="test_value"):
    """Spatial join test points/features to regions."""
    gdf = region_gdf[[region_id_col, "geometry"]].copy()

    geom_type = test_gdf.geometry.geom_type.iloc[0]
    if geom_type == "Point":
        joined = gpd.sjoin(test_gdf, gdf, how="inner", predicate="within")
    else:
        test_c = test_gdf.copy()
        test_c["geometry"] = test_c.geometry.centroid
        joined = gpd.sjoin(test_c, gdf, how="inner", predicate="within")

    result = pd.DataFrame({
        "point_idx": range(len(joined)),
        "region": joined[region_id_col].values,
        "value": joined[value_col].values,
    }).dropna()
    return result


def run_analysis(test_gdf, raster_path, regions, region_id_cols,
                 raster_band=1, iqr_factor=1.5):
    """Run the full neighbor-contrast analysis for all region schemes.

    Returns:
        neighbor_diffs: dict {name: np.array of |diff| values}
        scheme_points:  dict {name: DataFrame with point_idx, region, value}
    """
    neighbor_diffs = {}
    scheme_points = {}

    for name, region_gdf in regions.items():
        rid_col = region_id_cols[name]
        print(f"\n{'='*60}")
        print(f"  Processing: {name}")
        print(f"{'='*60}")

        # 1. Polygon adjacency
        adjacency = _build_polygon_adjacency(region_gdf, rid_col)
        n_adj = sum(len(v) for v in adjacency.values()) // 2
        print(f"  Adjacency: {len(adjacency)} regions, {n_adj} neighbor pairs")

        # 2. Spatial join
        if raster_path is not None:
            from rasterstats import zonal_stats
            zs = zonal_stats(region_gdf, raster_path, band=raster_band,
                             stats=["mean"], nodata=np.nan)
            pts = pd.DataFrame({
                "point_idx": range(len(region_gdf)),
                "region": region_gdf[rid_col].values,
                "value": [z["mean"] for z in zs],
            }).dropna()
        else:
            pts = _spatial_join_points(test_gdf, region_gdf, rid_col)

        scheme_points[name] = pts
        print(f"  Joined points: {len(pts)}")

        # 3. Pairwise |diff| for points in neighboring regions
        region_to_points = pts.groupby("region").apply(
            lambda g: g[["point_idx", "value"]].values.tolist()
        ).to_dict()

        diffs = []
        for region_a, neighbors in adjacency.items():
            if region_a not in region_to_points:
                continue
            points_a = region_to_points[region_a]
            for region_b in neighbors:
                if region_b not in region_to_points:
                    continue
                if str(region_a) >= str(region_b):
                    continue
                points_b = region_to_points[region_b]
                for _, val_a in points_a:
                    for _, val_b in points_b:
                        diffs.append(abs(val_a - val_b))

        diffs = np.array(diffs)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            diffs = remove_outliers_iqr(diffs, factor=iqr_factor)

        neighbor_diffs[name] = diffs
        print(f"  Neighbor differences: {len(diffs)} pairs (after IQR filtering)")
        if len(diffs) > 0:
            print(f"  Mean |diff|: {np.mean(diffs):.4f}")

    print(f"\n{'='*60}")
    print("All schemes processed.")
    return neighbor_diffs, scheme_points


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Statistical tests
# ═══════════════════════════════════════════════════════════════════════

def run_mann_whitney(neighbor_diffs, reference="Regionalization Output"):
    """Run Mann-Whitney U tests: Other < Reference (alternative='less').

    If p < 0.05: the other scheme has lower diffs → Reference is better.

    Returns:
        mw_results: dict {other_name: {"U": float, "p": float, "significant": bool}}
    """
    ref_diffs = neighbor_diffs.get(reference, np.array([]))
    comparison_names = [n for n in neighbor_diffs if n != reference]
    mw_results = {}

    print(f"Mann-Whitney U Test (H_a: Other < {reference})\n")
    print(f"{'Comparison':<35s} {'U-stat':>12s} {'p-value':>12s} {reference + ' better?':>20s}")
    print("-" * 82)

    for other_name in comparison_names:
        other_diffs = neighbor_diffs.get(other_name, np.array([]))
        if len(ref_diffs) == 0 or len(other_diffs) == 0:
            print(f"{other_name + ' vs ' + reference:<35s} {'N/A':>12s} {'N/A':>12s} {'N/A':>20s}")
            continue

        stat, p = mannwhitneyu(other_diffs, ref_diffs, alternative='less')
        sig = "Yes (p<0.05)" if p < 0.05 else "No"
        mw_results[other_name] = {"U": stat, "p": p, "significant": p < 0.05}
        print(f"{other_name + ' vs ' + reference:<35s} {stat:>12.2f} {p:>12.4f} {sig:>20s}")

    print(f"\nIf p < 0.05: the other scheme has lower diffs → {reference} is better.")
    return mw_results


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_cdf(neighbor_diffs, mw_results, reference="Regionalization Output"):
    """CDF comparison plots: each other scheme vs. Reference."""
    ref_diffs = neighbor_diffs.get(reference, np.array([]))
    comparison_names = [n for n in neighbor_diffs if n != reference]

    line_styles = ['-', '--']
    colors = ['brown', 'orange']
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    n = len(comparison_names)
    if n == 0:
        print("No comparisons to plot.")
        return

    fig, axs = plt.subplots(1, n, figsize=(5 * n, 4), dpi=150)
    if n == 1:
        axs = [axs]

    for i, other_name in enumerate(comparison_names):
        ax = axs[i]
        other_diffs = neighbor_diffs.get(other_name, np.array([]))

        for j, (arr, lbl) in enumerate([(other_diffs, other_name), (ref_diffs, reference)]):
            if len(arr) == 0:
                continue
            kde = gaussian_kde(arr)
            x = np.linspace(arr.min(), arr.max(), 200)
            cdf = np.cumsum(kde(x))
            cdf /= cdf[-1]
            ax.plot(x, cdf, label=lbl, linestyle=line_styles[j], linewidth=2, color=colors[j])

        if other_name in mw_results:
            r = mw_results[other_name]
            ax.text(0.95, 0.05,
                    f"U={r['U']:.2f}\np={r['p']:.3f}\nAlternative: less",
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.6))

        prefix = labels[i] if i < len(labels) else ""
        ax.set_title(f"{prefix} {other_name} vs {reference}", fontsize=12, loc='left')
        ax.set_xlabel("Neighbor Value Difference", fontsize=11)
        ax.set_ylabel("CDF", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_bar_chart(neighbor_diffs, reference="Regionalization Output"):
    """Bar chart of mean neighbor |diff| per scheme (higher = better)."""
    names = list(neighbor_diffs.keys())
    means = [np.mean(d) if len(d) > 0 else 0 for d in neighbor_diffs.values()]
    colors = ["#2ecc71" if n == reference else "#3498db" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, means, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Mean Neighbor |Difference|", fontsize=12)
    ax.set_title("Mean Neighbor-Contrast by Regionalization Scheme\n(Higher = Better Separation)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.3 if means else 1)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_violin(neighbor_diffs):
    """Violin plot of neighbor difference distributions."""
    violin_data = []
    for name, d in neighbor_diffs.items():
        if len(d) > 0:
            violin_data.append(pd.DataFrame({"Regionalization": name, "Neighbor |diff|": d}))

    if not violin_data:
        print("No data to plot.")
        return

    violin_df = pd.concat(violin_data, ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=violin_df, x="Regionalization", y="Neighbor |diff|",
                   palette="Set2", inner="box", ax=ax)
    ax.set_title("Distribution of Neighbor-Contrast Differences\n(Higher = Sharper Boundaries)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Absolute Difference Between Neighboring Points")
    ax.set_xlabel("")
    sns.despine()
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Section 7: Comparison table & summary
# ═══════════════════════════════════════════════════════════════════════

def show_comparison_table(neighbor_diffs, scheme_points, regions,
                          region_id_cols, mw_results, reference="Regionalization Output"):
    """Display a styled comparison table."""
    rows = []
    for name in regions:
        d = neighbor_diffs.get(name, np.array([]))
        row = {
            "Regionalization": name,
            "Mean |diff|": f"{np.mean(d):.4f}" if len(d) > 0 else "N/A",
            "Median |diff|": f"{np.median(d):.4f}" if len(d) > 0 else "N/A",
            "N pairs": len(d),
            "N regions": regions[name][region_id_cols[name]].nunique(),
            "N test points": len(scheme_points.get(name, [])),
        }
        if name != reference and name in mw_results:
            r = mw_results[name]
            row[f"vs {reference} (U)"] = f"{r['U']:.2f}"
            row[f"vs {reference} (p)"] = f"{r['p']:.4f}"
            row[f"{reference} better?"] = "Yes" if r["significant"] else "No"
        else:
            row[f"vs {reference} (U)"] = "-"
            row[f"vs {reference} (p)"] = "-"
            row[f"{reference} better?"] = "(reference)" if name == reference else "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)

    def highlight_highest(s):
        vals = []
        for v in s:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(float("-inf"))
        is_max = [v == max(vals) for v in vals]
        return ["background-color: #d4edda; font-weight: bold" if m else "" for m in is_max]

    styled = df.style.apply(
        highlight_highest, subset=["Mean |diff|"], axis=0
    ).set_caption("Neighbor-Contrast Comparison (Higher |diff| = Better Separation)")
    display(styled)
    return df


def print_summary(neighbor_diffs, mw_results, reference="Regionalization Output"):
    """Print auto-generated interpretation summary."""
    if not neighbor_diffs:
        print("No results available.")
        return

    ranked = sorted(
        [(n, np.mean(d)) for n, d in neighbor_diffs.items() if len(d) > 0],
        key=lambda x: x[1], reverse=True
    )
    best_name, best_mean = ranked[0]
    worst_name, worst_mean = ranked[-1]

    print("=" * 70)
    print("  INTERPRETATION SUMMARY (Neighbor-Contrast Method)")
    print("=" * 70)
    print()
    print(f"  Best regionalization:  {best_name}")
    print(f"    Mean neighbor |diff| = {best_mean:.4f}")
    print(f"    Boundaries create the sharpest contrasts — regions are well-separated.")
    print()
    print(f"  Worst regionalization: {worst_name}")
    print(f"    Mean neighbor |diff| = {worst_mean:.4f}")
    print(f"    Boundaries create weak contrasts — regions are poorly separated.")
    print()

    ref_mean = np.mean(neighbor_diffs.get(reference, [0]))
    print(f"  Pairwise significance vs {reference}:")
    for other_name, r in mw_results.items():
        other_mean = np.mean(neighbor_diffs.get(other_name, [0]))
        pct = ((ref_mean - other_mean) / other_mean * 100) if other_mean > 0 else 0
        if r["significant"]:
            print(f"    vs {other_name}: {reference} is {pct:.1f}% higher "
                  f"(p={r['p']:.4f}) — SIGNIFICANT")
        else:
            print(f"    vs {other_name}: difference not significant (p={r['p']:.4f})")

    print()
    print("  Interpretation:")
    print("    Higher mean |diff| = region boundaries create sharper contrasts.")
    print("    Mann-Whitney U (alternative='less') tests whether the other scheme")
    print(f"    has lower diffs than {reference}. If p < 0.05, {reference} is better.")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# Section 8: Export
# ═══════════════════════════════════════════════════════════════════════

def export_results(comparison_df, neighbor_diffs, regions):
    """Export comparison table and raw diffs as CSVs. Trigger Colab download."""
    comparison_df.to_csv("neighbor_contrast_comparison.csv", index=False)
    print("Saved: neighbor_contrast_comparison.csv")

    for name in regions:
        d = neighbor_diffs.get(name, np.array([]))
        if len(d) > 0:
            safe = name.replace(" ", "_").lower()
            pd.DataFrame({"neighbor_diff": d}).to_csv(f"neighbor_diffs_{safe}.csv", index=False)
            print(f"Saved: neighbor_diffs_{safe}.csv")

    try:
        from google.colab import files as colab_files
        colab_files.download("neighbor_contrast_comparison.csv")
        for name in regions:
            d = neighbor_diffs.get(name, np.array([]))
            if len(d) > 0:
                safe = name.replace(" ", "_").lower()
                colab_files.download(f"neighbor_diffs_{safe}.csv")
    except ImportError:
        print("\n(Not running in Colab — files saved to working directory.)")
