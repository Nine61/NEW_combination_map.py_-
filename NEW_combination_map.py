"""
UAV 시뮬레이션용 지형 및 도심지 맵 데이터 통합 생성 스크립트
- SRTM 위성 데이터를 활용한 산악 지형(고도/저지대) 생성
- OSMnx GIS 데이터를 활용한 도심지(건물 장애물) 생성
- 모든 출력 맵은 알고리즘 환경에 맞게 700x700 2D Grid(.npy)로 규격화됨
"""

import numpy as np
import scipy.ndimage
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tifffile
import osmnx as ox
from shapely.geometry import Point
import requests

# ==========================================
# 1. 공통 환경 및 경로 설정
# ==========================================
# BASE_DIR: 현재 이 파이썬 파일이 위치한 절대 경로를 자동으로 추적
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TAREGET_SIZE: 드론 자율비행 강화학습/경로탐색 알고리즘이 입력받을 최종 맵의 해상도(State Space 크기)
TAREGET_SIZE = 700
# 생성된 .npy 파일들이 저장될 data 폴더의 절대 경로 지정
SAVE_DIR = os.path.join(BASE_DIR, "data") 
FALLBACK_API_KEY = "19c8ec4f172050fed3736d53c0efd6c0"

# ==========================================
# 2. 산악 지형(SRTM) 파라미터 (북한산 일대)
# ==========================================
TARGET_LAT1 = 37.6587  # 북한산 중심 위도
TARGET_LON1 = 126.9780 # 북한산 중심 경도

# 💡 하이브리드 맵 생성을 위한 임무 환경(Mission Environment) 조건
ALTITUDE_LIMIT = 500   # 거대 장애물(산맥 등) 기준 고도. 드론의 최대 상승 한계를 고려하여 설정.
MICRO_DENSITY  = 0.01  # 평지 내 돌발 미세 장애물(송전탑, 조류 등) 발생 확률 (1%)
LOWLAND_PERCENTILE = 10  # 전체 지형 중 하위 10%를 '저지대(Lowland)'로 분류. 은폐/엄폐 기동 경로로 활용.

# ==========================================
# 3. 도심 지형(OSMnx) 파라미터 (광운대 일대)
# ==========================================
TARGET_LAT2 = 37.6194  # 광운대 중심 위도
TARGET_LON2 = 127.0598 # 광운대 중심 경도
RADIUS_M = 200       # 중심 좌표로부터 데이터를 추출할 반경 (미터 단위)
GRID_SIZE = 700      # Rasterization(벡터->그리드) 시 쪼갤 해상도. TAREGET_SIZE와 동일하게 맞춤.
MIN_BUILDING_AREA = 100  # 최소 건물 면적(m^2). 드론 비행에 지장을 주지 않는 가건물이나 작은 구조물을 필터링.


# ==========================================
# 4. 기능 함수 정의 (공장)
# ==========================================

def crop_quadrant(terrain, quadrant="SW"):
    """
    광범위한 DEM(디지털 표고 모델) 행렬 데이터를 4개의 임무 구역(Sector)으로 분할하는 함수
    """
    h, w = terrain.shape
    quadrant_map = {
        "NW": terrain[:h//2, :w//2], "NE": terrain[:h//2, w//2:],
        "SW": terrain[h//2:, :w//2], "SE": terrain[h//2:, w//2:],
    }
    # 요청된 구역이 없으면 기본값으로 SW(남서) 구역 반환
    return quadrant_map.get(quadrant, quadrant_map["SW"])

def fetch_real_srtm_data(lat, lon, size_km=3):
    """
    OpenTopography API를 호출하여 실제 지구의 표면 고도(SRTM) 데이터를 다운로드하는 함수
    """
    api_key = os.environ.get("SRTM_API_KEY", FALLBACK_API_KEY).strip()
    if not api_key: return None
    
    # 킬로미터(km) 단위를 위경도 도(degree) 단위로 대략 변환
    lat_delta, lon_delta = (size_km / 2) / 111, (size_km / 2) / 88
    url = f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south={lat-lat_delta}&north={lat+lat_delta}&west={lon-lon_delta}&east={lon+lon_delta}&outputFormat=GTiff&API_Key={api_key}"
    
    # 다운로드 받은 임시 tif 파일의 안전한 절대 경로
    temp_file_path = os.path.join(BASE_DIR, "temp.tif")

    print("🌐 지형 데이터 요청 중...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(temp_file_path, "wb") as f: 
                f.write(response.content)
            # tif 파일을 읽어 2차원 float 행렬로 변환
            terrain = tifffile.imread(temp_file_path)
            os.remove(temp_file_path) # 사용 후 임시 파일 삭제
            return terrain.astype(np.float32)
        else:
            print(f"⚠️ API 요청 실패: 상태 코드 {response.status_code}")
    except Exception as e: 
        print(f"⚠️ 오류: {e}")
    return None

def build_hybrid_map(terrain, altitude_limit, micro_density):
    """
    연속적인 고도 값(실수)을 가진 terrain 행렬을 
    드론 알고리즘이 인식할 수 있는 이산적인 상태 공간(0, 1, 2)으로 범주화하는 함수
    """
    # 1. 저지대(Lowland) 추출: 백분위수(Percentile)를 이용해 하위 10% 높이의 임계값 계산
    lowland_threshold = np.percentile(terrain, LOWLAND_PERCENTILE)
    lowland_map = (terrain < lowland_threshold).astype(int)

    # 2. 거대 장애물(Macro) 추출: 비행 제한 고도를 초과하는 구역
    macro_map = (terrain > altitude_limit).astype(int)
    
    # 3. 미세 장애물(Micro) 추출: 난수 행렬을 생성하여 평지와 저지대가 아닌 곳에만 밀도만큼 뿌림
    rand_matrix = np.random.rand(*terrain.shape)
    micro_map = ((rand_matrix < micro_density) & (macro_map == 0) & (lowland_map == 0)).astype(int)

    # 4. 최종 하이브리드 맵 병합 (0: 안전 지면, 1: 장애물, 2: 저지대)
    hybrid_map = np.zeros_like(terrain, dtype=int)
    hybrid_map[macro_map == 1] = 1
    hybrid_map[micro_map == 1] = 1
    hybrid_map[lowland_map == 1] = 2  
    
    print(f"📊 지형 셀 통계:")
    print(f"  └ 🟫 일반 지면(0): {(hybrid_map == 0).sum()}칸")
    print(f"  └ 🚧 장애물(1): {(hybrid_map == 1).sum()}칸")
    print(f"  └ 🟨 저지대(2): {(hybrid_map == 2).sum()}칸")
    return hybrid_map

def _resize_map(terrain_map, target_size = TAREGET_SIZE):
    """
    잘려진 행렬 데이터를 알고리즘 요구 규격(700x700)으로 스케일링하는 헬퍼 함수
    """
    current_h, current_w = terrain_map.shape
    zoom_factor_h = target_size / current_h
    zoom_factor_w = target_size / current_w

    # order=0 (Nearest-neighbor interpolation)을 사용하여 0, 1, 2 값이 섞여서 0.5 같은 소수가 되지 않도록 방지
    resized_map = scipy.ndimage.zoom(terrain_map, (zoom_factor_h, zoom_factor_w), order=0)
    return resized_map


# ==========================================
# 5. 메인 통합 함수 (작업장)
# ==========================================

def generate_enivronment_map(map_type):
    """
    UI 또는 메인 루프에서 호출받아 특정 구역의 시뮬레이션 환경(.npy)을 생성하는 메인 파이프라인
    - map_type: 'SW','SE','NW','NE'(북한산), 'KWU'(광운대)
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    result_map = None
    
    # ----------------------------------------
    # [A] 산악 지형 처리 로직 (SRTM)
    # ----------------------------------------
    if map_type in ['SW','SE','NW','NE']:
        print(f"\n⛰️ 북한산 {map_type} 지형 생성 중...")
        terrain = fetch_real_srtm_data(TARGET_LAT1,TARGET_LON1)
        
        # 네트워크 오류 등으로 데이터를 가져오지 못했을 때의 방어(Fail-safe) 로직
        if terrain is None:
            print("❌ 지형 데이터를 가져오지 못해 맵 생성을 건너뜁니다.")
            return None

        croopped_terrain = crop_quadrant(terrain, map_type)
        raw_hybrid_map = build_hybrid_map(croopped_terrain, ALTITUDE_LIMIT, MICRO_DENSITY)
        
        # 알고리즘 입력 규격(700x700)에 맞게 리사이징
        result_map = _resize_map(raw_hybrid_map,TAREGET_SIZE)
    
    # ----------------------------------------
    # [B] 도심 지형 처리 로직 (OSMnx)
    # ----------------------------------------
    elif map_type == 'KWU':
        print(f"\n🏫 광운대학교 도심지 {map_type} 맵 생성 중 (작은 건물 필터링 적용)...")
        tags = {'building': True}
        
        # 1. OSMnx를 통해 벡터(Vector) 다각형 데이터 다운로드
        buildings = ox.features_from_point((TARGET_LAT2, TARGET_LON2), tags=tags, dist=RADIUS_M)
        
        # 2. 면적 필터링을 위한 UTM(평면) 좌표계 투영 (위경도 상태에서는 정확한 제곱미터 계산 불가)
        buildings_proj = ox.projection.project_gdf(buildings) 
        
        # 3. 드론 회피 기동에 유의미한 거점(면적 MIN_BUILDING_AREA 이상) 건물만 추출
        buildings_proj = buildings_proj[buildings_proj.geometry.area >= MIN_BUILDING_AREA]
        
        # 4. 필터링된 데이터를 다시 지리(위경도) 좌표계로 복원
        buildings = ox.projection.project_gdf(buildings_proj, to_latlong=True)

        if buildings.empty:
            print("⚠️ 설정한 기준을 만족하는 건물이 없습니다. MIN_BUILDING_AREA를 낮춰보세요.")
            return None

        # 5. 수백 개의 건물 다각형을 하나의 거대한 MultiPolygon으로 병합 (교차 연산 속도 최적화)
        unified_buildings = buildings.geometry.unary_union
        
        # 6. 추출 영역의 동서남북 바운더리 박스 획득
        west, south, east, north = buildings.total_bounds

        # 7. 바운더리를 700x700 픽셀(격자)로 나누기 위한 좌표 리스트 생성
        x_coords = np.linspace(west, east, GRID_SIZE)
        y_coords = np.linspace(north, south, GRID_SIZE) # Y축은 북쪽에서 남쪽으로 배열 인덱싱과 방향을 맞춤

        urban_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        
        # 8. 래스터화(Rasterization): 700x700개의 각 좌표 점이 건물 덩어리 안에 포함되는지 검사
        for y_idx, y in enumerate(y_coords):
            for x_idx, x in enumerate(x_coords):
                point = Point(x, y)
                if unified_buildings.intersects(point):
                    urban_map[y_idx, x_idx] = 1  # 건물이 존재하면 1 (장애물) 할당
                    
        result_map = urban_map
        
    else:
        print("오류: 잘못된 map_type이 입력되었습니다.")
        return None
        
    # ----------------------------------------
    # [C] 결과물 저장
    # ----------------------------------------
    save_path = os.path.join(SAVE_DIR, f"{map_type}_map.npy")
    np.save(save_path, result_map)
    print(f"✅ 저장 완료! 파일 경로: {save_path} (크기: {result_map.shape})")

    return result_map


# ==========================================
# 6. 실행 트리거
# ==========================================
if __name__ == "__main__":
     #이 스크립트를 직접 실행할 경우, 5가지 맵을 순차적으로 모두 생성하여 data 폴더에 저장함
    for map_type in ['SW', 'SE', 'NW', 'NE', 'KWU']:
        generate_enivronment_map(map_type)
