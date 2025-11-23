# streamlit_app.py
# 실행: streamlit run --server.port 3000 --server.address 0.0.0.0 streamlit_app.py

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
import streamlit as st

# 🔵 Cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 🔤 한글 폰트 (Pretendard-Bold.ttf)
from matplotlib import font_manager as fm, rcParams
from pathlib import Path
font_path = Path("fonts/Pretendard-Bold.ttf").resolve()
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))
    rcParams["font.family"] = font_prop.get_name()
else:
    font_prop = fm.FontProperties()
rcParams["axes.unicode_minus"] = False

# -------------------------------------------------
# ✅ ERDDAP: SOEST Hawaii 인스턴스 한 곳만 사용 (고정)
#   - OISST v2.1 (AVHRR) anomaly 포함
#   - 이 인스턴스는 현재 2024-12-31까지 제공됨
# -------------------------------------------------
ERDDAP_URL = "https://erddap.aoml.noaa.gov/hdb/erddap/griddap/SST_OI_DAILY_1981_PRESENT_T"

def _open_ds(url_base: str):
    """서버 설정에 따라 .nc 필요할 수 있어 두 번 시도 (동일 엔드포인트 고정)."""
    try:
        return xr.open_dataset(url_base, decode_times=True)
    except Exception:
        return xr.open_dataset(url_base + ".nc", decode_times=True)

def _standardize_anom_field(ds: xr.Dataset, target_time: pd.Timestamp) -> xr.DataArray:
    """
    - 변수: 'anom'
    - 깊이 차원(있다면): 표층 선택
    - 좌표명: latitude/longitude → lat/lon 통일
    - 시간: 데이터 커버리지 바깥이면 경계로 클램프 후 'nearest'
    """
    da = ds["anom"]

    # 깊이 차원 표층 선택
    for d in ["zlev", "depth", "lev"]:
        if d in da.dims:
            da = da.sel({d: da[d].values[0]})
            break

    # 시간 클램프 + nearest (멀리 점프 방지)
    times = pd.to_datetime(ds["time"].values)
    tmin, tmax = times.min(), times.max()
    if target_time < tmin:
        target_time = tmin
    elif target_time > tmax:
        target_time = tmax
    da = da.sel(time=target_time, method="nearest").squeeze(drop=True)

    # 좌표명 통일
    rename_map = {}
    if "latitude" in da.coords:  rename_map["latitude"]  = "lat"
    if "longitude" in da.coords: rename_map["longitude"] = "lon"
    if rename_map:
        da = da.rename(rename_map)

    return da

# -----------------------------
# 데이터 접근 (SOEST만 사용)
# -----------------------------
@st.cache_data(show_spinner=False)
def list_available_times() -> pd.DatetimeIndex:
    ds = _open_ds(ERDDAP_URL)
    times = pd.to_datetime(ds["time"].values)
    ds.close()
    return pd.DatetimeIndex(times)

@st.cache_data(show_spinner=True)
def load_anomaly(date: pd.Timestamp, bbox=None) -> xr.DataArray:
    """
    선택 날짜의 anomaly(°C) 2D 필드 반환.
    bbox=(lat_min, lat_max, lon_min, lon_max); 경도 -180~180.
    날짜 변경선 횡단 시 자동 분할-결합.
    """
    ds = _open_ds(ERDDAP_URL)
    da = _standardize_anom_field(ds, date)

    # bbox 슬라이스
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox

        # 위도
        if lat_min <= lat_max:
            da = da.sel(lat=slice(lat_min, lat_max))
        else:
            da = da.sel(lat=slice(lat_max, lat_min))

        # 경도 (+ 날짜변경선 처리)
        if lon_min <= lon_max:
            da = da.sel(lon=slice(lon_min, lon_max))
        else:
            left  = da.sel(lon=slice(lon_min, 180))
            right = da.sel(lon=slice(-180, lon_max))
            da = xr.concat([left, right], dim="lon")

    ds.close()
    return da

# -----------------------------
# Cartopy Plot
# -----------------------------
def plot_cartopy_anomaly(
    da: xr.DataArray,
    title: str,
    vabs: float = 5.0,
    projection=ccrs.Robinson(),
    extent=None,
):
    fig = plt.figure(figsize=(12.5, 6.5))
    ax = plt.axes(projection=projection)

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=3)

    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    cmap = cm.get_cmap("RdBu_r").copy()
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    if "lon" in da.coords:
        da = da.sortby("lon")

    im = ax.pcolormesh(
        da["lon"], da["lat"], da.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, shading="auto", zorder=2
    )

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.04, shrink=0.9)
    cbar.set_label("해수면 온도 편차 (°C, 1971–2000 기준)", fontproperties=font_prop)

    ax.set_title(title, pad=8, fontproperties=font_prop)
    fig.tight_layout()
    return fig

# -----------------------------
# UI
# -----------------------------
st.sidebar.header("🛠️ 보기 옵션")

# 날짜 범위 = SOEST 실제 커버리지로 제한
with st.spinner("사용 가능한 날짜 불러오는 중..."):
    times = list_available_times()
tmin, tmax = times.min().date(), times.max().date()

# ✅ 기본 시작일 = 2024-08-15 (커버리지 범위 바깥이면 자동 조정)
DEFAULT_START = pd.Timestamp("2024-08-15")
if DEFAULT_START.date() < tmin:
    default_date = times[0]
elif DEFAULT_START.date() > tmax:
    default_date = times[-1]
else:
    default_date = DEFAULT_START

date = st.sidebar.date_input(
    "날짜 선택",
    value=default_date.date(),
    min_value=tmin,
    max_value=tmax,
)
date = pd.Timestamp(date)

# 영역 프리셋
preset = st.sidebar.selectbox(
    "영역 선택",
    [
        "전 지구",
        "동아시아(한국 포함)",
종
    </div>
    """,
    unsafe_allow_html=True
)
