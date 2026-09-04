import math
from typing import Any
from fastapi import APIRouter, HTTPException, Query

from app.ml.predictor import predictor
from app.services.feature_service import FeatureService

router = APIRouter(prefix="/api/locations", tags=["locations"])

# Comprehensive Northeast India Locations Database stored in the backend
NORTHEAST_LOCATIONS_DB: list[dict[str, Any]] = [
    # --- ARUNACHAL PRADESH ---
    {
        "id": "tawang",
        "name": "Tawang",
        "state": "Arunachal Pradesh",
        "district": "Tawang",
        "coordinates": [27.5861, 91.8660],
        "elevation_m": 3048,
        "slope_degrees": 41,
        "aspect_degrees": 145,
        "rainfall_24h": 92,
        "rainfall_3d": 178,
        "rainfall_7d": 260,
        "soil_moisture": 0.78,
        "description": "Colluvial debris on steep hill slopes above township road; high saturation from prolonged rainfall.",
        "evacuation_center": "Tawang Town Hall Community Shelter",
        "shelter_distance": "1.2 km away",
        "helpline": "03794-222225 (DDMA Tawang)",
        "sensors_count": 6,
        "population_at_risk": 1420,
    },
    {
        "id": "sela-pass",
        "name": "Sela Pass",
        "state": "Arunachal Pradesh",
        "district": "West Kameng",
        "coordinates": [27.5050, 92.1030],
        "elevation_m": 4170,
        "slope_degrees": 46,
        "aspect_degrees": 180,
        "rainfall_24h": 104,
        "rainfall_3d": 210,
        "rainfall_7d": 310,
        "soil_moisture": 0.84,
        "description": "High-altitude glacial scree & talus slopes abutting strategic NH-13 corridor.",
        "evacuation_center": "BRO 42nd Task Force Staging Ground",
        "shelter_distance": "2.1 km away",
        "helpline": "1078 (Disaster Toll-Free)",
        "sensors_count": 8,
        "population_at_risk": 380,
    },
    {
        "id": "bomdila",
        "name": "Bomdila",
        "state": "Arunachal Pradesh",
        "district": "West Kameng",
        "coordinates": [27.2645, 92.4244],
        "elevation_m": 2217,
        "slope_degrees": 34,
        "aspect_degrees": 120,
        "rainfall_24h": 68,
        "rainfall_3d": 132,
        "rainfall_7d": 195,
        "soil_moisture": 0.65,
        "description": "Weathered phyllite bedrock with hillside terrace settlements and roadside toe erosion.",
        "evacuation_center": "Bomdila Govt Higher Secondary Safe Zone",
        "shelter_distance": "0.8 km away",
        "helpline": "03782-222123",
        "sensors_count": 5,
        "population_at_risk": 950,
    },
    {
        "id": "dirang",
        "name": "Dirang",
        "state": "Arunachal Pradesh",
        "district": "West Kameng",
        "coordinates": [27.3592, 92.2389],
        "elevation_m": 1560,
        "slope_degrees": 29,
        "aspect_degrees": 95,
        "rainfall_24h": 54,
        "rainfall_3d": 98,
        "rainfall_7d": 145,
        "soil_moisture": 0.58,
        "description": "Kameng river terrace slopes with moderate clay loam saturation.",
        "evacuation_center": "Dirang Community Hall",
        "shelter_distance": "1.5 km away",
        "helpline": "03782-242233",
        "sensors_count": 3,
        "population_at_risk": 420,
    },
    {
        "id": "lumla",
        "name": "Lumla",
        "state": "Arunachal Pradesh",
        "district": "Tawang",
        "coordinates": [27.5333, 91.7167],
        "elevation_m": 2400,
        "slope_degrees": 38,
        "aspect_degrees": 160,
        "rainfall_24h": 88,
        "rainfall_3d": 164,
        "rainfall_7d": 235,
        "soil_moisture": 0.75,
        "description": "Sheared mica schist slope vulnerable to debris slides during continuous downpours.",
        "evacuation_center": "Lumla Sub-Divisional Officer Complex",
        "shelter_distance": "0.6 km away",
        "helpline": "03794-273210",
        "sensors_count": 4,
        "population_at_risk": 610,
    },
    {
        "id": "itanagar",
        "name": "Itanagar",
        "state": "Arunachal Pradesh",
        "district": "Papum Pare",
        "coordinates": [27.0844, 93.6053],
        "elevation_m": 750,
        "slope_degrees": 22,
        "aspect_degrees": 110,
        "rainfall_24h": 38,
        "rainfall_3d": 72,
        "rainfall_7d": 110,
        "soil_moisture": 0.44,
        "description": "Tertiary sandstone foothills with localized urban drainage cut slope failures.",
        "evacuation_center": "Indira Gandhi Park Emergency Pavilion",
        "shelter_distance": "1.8 km away",
        "helpline": "0360-2212374 (State EOC)",
        "sensors_count": 4,
        "population_at_risk": 520,
    },
    {
        "id": "naharlagun",
        "name": "Naharlagun",
        "state": "Arunachal Pradesh",
        "district": "Papum Pare",
        "coordinates": [27.1067, 93.6931],
        "elevation_m": 480,
        "slope_degrees": 18,
        "aspect_degrees": 85,
        "rainfall_24h": 35,
        "rainfall_3d": 65,
        "rainfall_7d": 95,
        "soil_moisture": 0.41,
        "description": "Dikrong river basin margin with stable terrace gradients.",
        "evacuation_center": "TRIHMS Hospital Safe Ground",
        "shelter_distance": "1.0 km away",
        "helpline": "0360-2244256",
        "sensors_count": 2,
        "population_at_risk": 180,
    },
    {
        "id": "pasighat",
        "name": "Pasighat",
        "state": "Arunachal Pradesh",
        "district": "East Siang",
        "coordinates": [28.0667, 95.3333],
        "elevation_m": 153,
        "slope_degrees": 14,
        "aspect_degrees": 130,
        "rainfall_24h": 42,
        "rainfall_3d": 85,
        "rainfall_7d": 130,
        "soil_moisture": 0.48,
        "description": "Siang river debouchment plain with river toe scour and flash flood margins.",
        "evacuation_center": "Pasighat General Hospital Complex",
        "shelter_distance": "1.4 km away",
        "helpline": "0368-2222224",
        "sensors_count": 3,
        "population_at_risk": 310,
    },
    {
        "id": "ziro",
        "name": "Ziro",
        "state": "Arunachal Pradesh",
        "district": "Lower Subansiri",
        "coordinates": [27.5583, 93.8389],
        "elevation_m": 1688,
        "slope_degrees": 24,
        "aspect_degrees": 170,
        "rainfall_24h": 48,
        "rainfall_3d": 94,
        "rainfall_7d": 140,
        "soil_moisture": 0.52,
        "description": "Surrounding pine-clad ridge slopes encircling high valley plateau.",
        "evacuation_center": "Ziro Circuit House Safe Grounds",
        "shelter_distance": "1.1 km away",
        "helpline": "03788-224255",
        "sensors_count": 3,
        "population_at_risk": 290,
    },
    {
        "id": "aalo",
        "name": "Aalo",
        "state": "Arunachal Pradesh",
        "district": "West Siang",
        "coordinates": [28.1694, 94.8028],
        "elevation_m": 610,
        "slope_degrees": 31,
        "aspect_degrees": 105,
        "rainfall_24h": 62,
        "rainfall_3d": 118,
        "rainfall_7d": 175,
        "soil_moisture": 0.62,
        "description": "Yomgo river gorge hill slopes prone to rotational slips.",
        "evacuation_center": "Aalo Govt College Indoor Stadium",
        "shelter_distance": "1.3 km away",
        "helpline": "03783-222221",
        "sensors_count": 3,
        "population_at_risk": 740,
    },
    {
        "id": "roing",
        "name": "Roing",
        "state": "Arunachal Pradesh",
        "district": "Lower Dibang Valley",
        "coordinates": [28.1367, 95.8394],
        "elevation_m": 390,
        "slope_degrees": 26,
        "aspect_degrees": 125,
        "rainfall_24h": 58,
        "rainfall_3d": 112,
        "rainfall_7d": 168,
        "soil_moisture": 0.59,
        "description": "Dibang foothill boundary fault zone subject to intense debris flows.",
        "evacuation_center": "Roing Multi-Purpose Community Hall",
        "shelter_distance": "0.9 km away",
        "helpline": "03803-222222",
        "sensors_count": 3,
        "population_at_risk": 480,
    },

    # --- SIKKIM ---
    {
        "id": "gangtok",
        "name": "Gangtok",
        "state": "Sikkim",
        "district": "East Sikkim",
        "coordinates": [27.3389, 88.6065],
        "elevation_m": 1650,
        "slope_degrees": 36,
        "aspect_degrees": 140,
        "rainfall_24h": 74,
        "rainfall_3d": 145,
        "rainfall_7d": 215,
        "soil_moisture": 0.72,
        "description": "Chongey & 9th Mile active slide belts on mica-gneiss bedrock dipping toward valley floor.",
        "evacuation_center": "Paljor Stadium Indoor Sports Complex",
        "shelter_distance": "1.2 km away",
        "helpline": "03592-202726 (Sikkim State Disaster Mgt)",
        "sensors_count": 7,
        "population_at_risk": 2100,
    },
    {
        "id": "mangan",
        "name": "Mangan",
        "state": "Sikkim",
        "district": "North Sikkim",
        "coordinates": [27.5111, 88.5333],
        "elevation_m": 1310,
        "slope_degrees": 44,
        "aspect_degrees": 165,
        "rainfall_24h": 96,
        "rainfall_3d": 188,
        "rainfall_7d": 280,
        "soil_moisture": 0.81,
        "description": "Chronic slide corridor along Teesta river gorge; highly fractured Daling quartzite.",
        "evacuation_center": "Mangan DAC District Shelter & ITI Grounds",
        "shelter_distance": "0.9 km away",
        "helpline": "03592-234244",
        "sensors_count": 6,
        "population_at_risk": 1180,
    },
    {
        "id": "namchi",
        "name": "Namchi",
        "state": "Sikkim",
        "district": "South Sikkim",
        "coordinates": [27.1667, 88.3500],
        "elevation_m": 1315,
        "slope_degrees": 30,
        "aspect_degrees": 115,
        "rainfall_24h": 52,
        "rainfall_3d": 102,
        "rainfall_7d": 155,
        "soil_moisture": 0.57,
        "description": "Undulating ridge slopes with moderate stability along western flank.",
        "evacuation_center": "Bhaichung Stadium Safe Assembly Area",
        "shelter_distance": "1.5 km away",
        "helpline": "03595-254244",
        "sensors_count": 4,
        "population_at_risk": 620,
    },
    {
        "id": "geyzing",
        "name": "Geyzing",
        "state": "Sikkim",
        "district": "West Sikkim",
        "coordinates": [27.2833, 88.2500],
        "elevation_m": 1900,
        "slope_degrees": 33,
        "aspect_degrees": 135,
        "rainfall_24h": 60,
        "rainfall_3d": 116,
        "rainfall_7d": 170,
        "soil_moisture": 0.63,
        "description": "Hill town on spur with active downslope slip creep toward Rangit river.",
        "evacuation_center": "Geyzing Senior Secondary School Hall",
        "shelter_distance": "0.8 km away",
        "helpline": "03595-250888",
        "sensors_count": 4,
        "population_at_risk": 540,
    },
    {
        "id": "pelling",
        "name": "Pelling",
        "state": "Sikkim",
        "district": "West Sikkim",
        "coordinates": [27.3014, 88.2389],
        "elevation_m": 2150,
        "slope_degrees": 35,
        "aspect_degrees": 155,
        "rainfall_24h": 65,
        "rainfall_3d": 125,
        "rainfall_7d": 185,
        "soil_moisture": 0.67,
        "description": "High ridge slope with precipitation seepage affecting roadside retaining walls.",
        "evacuation_center": "Pelling Helipad & Community Center",
        "shelter_distance": "1.0 km away",
        "helpline": "03595-250888",
        "sensors_count": 3,
        "population_at_risk": 410,
    },
    {
        "id": "lachung",
        "name": "Lachung",
        "state": "Sikkim",
        "district": "North Sikkim",
        "coordinates": [27.6894, 88.7436],
        "elevation_m": 2700,
        "slope_degrees": 47,
        "aspect_degrees": 180,
        "rainfall_24h": 98,
        "rainfall_3d": 195,
        "rainfall_7d": 290,
        "soil_moisture": 0.83,
        "description": "Glacial valley walls prone to severe rockfall scree and debris torrents.",
        "evacuation_center": "Army Staging Post Safe Shelter",
        "shelter_distance": "1.4 km away",
        "helpline": "1078 (Disaster Helpline)",
        "sensors_count": 5,
        "population_at_risk": 720,
    },

    # --- MEGHALAYA ---
    {
        "id": "shillong",
        "name": "Shillong",
        "state": "Meghalaya",
        "district": "East Khasi Hills",
        "coordinates": [25.5788, 91.8933],
        "elevation_m": 1525,
        "slope_degrees": 31,
        "aspect_degrees": 125,
        "rainfall_24h": 64,
        "rainfall_3d": 128,
        "rainfall_7d": 188,
        "soil_moisture": 0.66,
        "description": "Shillong plateau boundary scarp with urban cut slope slips.",
        "evacuation_center": "JN Stadium Indoor Sports Complex Polo",
        "shelter_distance": "1.6 km away",
        "helpline": "0364-2225289",
        "sensors_count": 6,
        "population_at_risk": 1850,
    },
    {
        "id": "cherrapunji",
        "name": "Cherrapunji (Sohra)",
        "state": "Meghalaya",
        "district": "East Khasi Hills",
        "coordinates": [25.2700, 91.7300],
        "elevation_m": 1430,
        "slope_degrees": 43,
        "aspect_degrees": 175,
        "rainfall_24h": 185,
        "rainfall_3d": 410,
        "rainfall_7d": 680,
        "soil_moisture": 0.94,
        "description": "Massive orographic rainfall belt causing rapid pore pressure surges along cliff rims.",
        "evacuation_center": "Sohra Sub-Division Emergency Center",
        "shelter_distance": "0.7 km away",
        "helpline": "03637-234222",
        "sensors_count": 8,
        "population_at_risk": 1650,
    },
    {
        "id": "mawsynram",
        "name": "Mawsynram",
        "state": "Meghalaya",
        "district": "East Khasi Hills",
        "coordinates": [25.2972, 91.5833],
        "elevation_m": 1400,
        "slope_degrees": 41,
        "aspect_degrees": 170,
        "rainfall_24h": 192,
        "rainfall_3d": 430,
        "rainfall_7d": 710,
        "soil_moisture": 0.96,
        "description": "Highest rainfall zone on Earth with heavy joint seepage and sudden slope collapse.",
        "evacuation_center": "Mawsynram Block Safe Shelter",
        "shelter_distance": "0.8 km away",
        "helpline": "1078 (Disaster Helpline)",
        "sensors_count": 7,
        "population_at_risk": 1400,
    },
    {
        "id": "tura",
        "name": "Tura",
        "state": "Meghalaya",
        "district": "West Garo Hills",
        "coordinates": [25.5138, 90.2206],
        "elevation_m": 350,
        "slope_degrees": 32,
        "aspect_degrees": 130,
        "rainfall_24h": 70,
        "rainfall_3d": 138,
        "rainfall_7d": 202,
        "soil_moisture": 0.69,
        "description": "Tura Peak foothill slopes prone to debris flows along steep gully channels.",
        "evacuation_center": "Tura District Auditorium Safe Zone",
        "shelter_distance": "1.2 km away",
        "helpline": "03651-223835",
        "sensors_count": 4,
        "population_at_risk": 920,
    },
    {
        "id": "nongpoh",
        "name": "Nongpoh",
        "state": "Meghalaya",
        "district": "Ri-Bhoi",
        "coordinates": [25.9000, 91.8800],
        "elevation_m": 485,
        "slope_degrees": 30,
        "aspect_degrees": 100,
        "rainfall_24h": 62,
        "rainfall_3d": 120,
        "rainfall_7d": 178,
        "soil_moisture": 0.64,
        "description": "Strategic NH-6 highway hill cuttings with cut-slope failures during monsoon.",
        "evacuation_center": "Nongpoh Community Health Center Safe Zone",
        "shelter_distance": "1.3 km away",
        "helpline": "03638-232223",
        "sensors_count": 4,
        "population_at_risk": 780,
    },

    # --- ASSAM ---
    {
        "id": "haflong",
        "name": "Haflong",
        "state": "Assam",
        "district": "Dima Hasao",
        "coordinates": [25.1667, 93.0167],
        "elevation_m": 680,
        "slope_degrees": 39,
        "aspect_degrees": 150,
        "rainfall_24h": 86,
        "rainfall_3d": 170,
        "rainfall_7d": 252,
        "soil_moisture": 0.79,
        "description": "Severely dissected Barail range shale & sandstone; chronic sinking zone for railway & hill roads.",
        "evacuation_center": "Haflong Council Stadium Community Shelter",
        "shelter_distance": "0.9 km away",
        "helpline": "03673-236224 (DDMA Dima Hasao)",
        "sensors_count": 7,
        "population_at_risk": 2400,
    },
    {
        "id": "guwahati",
        "name": "Guwahati (Hill Slopes)",
        "state": "Assam",
        "district": "Kamrup Metro",
        "coordinates": [26.1445, 91.7362],
        "elevation_m": 120,
        "slope_degrees": 33,
        "aspect_degrees": 135,
        "rainfall_24h": 68,
        "rainfall_3d": 134,
        "rainfall_7d": 198,
        "soil_moisture": 0.68,
        "description": "Narakasur, Sarania, and Kharghuli hill slopes with high population density on steep cuts.",
        "evacuation_center": "Sarusajai Stadium Safe Complex",
        "shelter_distance": "2.5 km away",
        "helpline": "1077 (Kamrup Metro Disaster Control)",
        "sensors_count": 6,
        "population_at_risk": 3100,
    },
    {
        "id": "diphu",
        "name": "Diphu",
        "state": "Assam",
        "district": "Karbi Anglong",
        "coordinates": [25.8400, 93.4300],
        "elevation_m": 186,
        "slope_degrees": 23,
        "aspect_degrees": 120,
        "rainfall_24h": 46,
        "rainfall_3d": 90,
        "rainfall_7d": 135,
        "soil_moisture": 0.51,
        "description": "Low rolling hills of Mikir massif with moderate slope debris vulnerability.",
        "evacuation_center": "Diphu Club & Sports Indoor Hall",
        "shelter_distance": "1.1 km away",
        "helpline": "03671-272255",
        "sensors_count": 3,
        "population_at_risk": 410,
    },

    # --- NAGALAND ---
    {
        "id": "kohima",
        "name": "Kohima",
        "state": "Nagaland",
        "district": "Kohima",
        "coordinates": [25.6751, 94.1086],
        "elevation_m": 1444,
        "slope_degrees": 37,
        "aspect_degrees": 145,
        "rainfall_24h": 81,
        "rainfall_3d": 162,
        "rainfall_7d": 242,
        "soil_moisture": 0.77,
        "description": "Disang shale formation with notorious active sinking zones at Dzüdza & NH-29 corridor.",
        "evacuation_center": "Kohima Local Ground Indira Gandhi Stadium",
        "shelter_distance": "1.5 km away",
        "helpline": "0370-2291122 (NSDMA Kohima)",
        "sensors_count": 6,
        "population_at_risk": 2200,
    },
    {
        "id": "mokokchung",
        "name": "Mokokchung",
        "state": "Nagaland",
        "district": "Mokokchung",
        "coordinates": [26.3200, 94.5200],
        "elevation_m": 1325,
        "slope_degrees": 32,
        "aspect_degrees": 130,
        "rainfall_24h": 63,
        "rainfall_3d": 122,
        "rainfall_7d": 180,
        "soil_moisture": 0.65,
        "description": "Long anticlinal hill ridge with high weathering of sandstone along spur roads.",
        "evacuation_center": "Mokokchung Town Hall Complex",
        "shelter_distance": "0.8 km away",
        "helpline": "0369-2226233",
        "sensors_count": 4,
        "population_at_risk": 780,
    },

    # --- MANIPUR ---
    {
        "id": "tamenglong",
        "name": "Tamenglong",
        "state": "Manipur",
        "district": "Tamenglong",
        "coordinates": [24.9800, 93.5000],
        "elevation_m": 1260,
        "slope_degrees": 42,
        "aspect_degrees": 165,
        "rainfall_24h": 90,
        "rainfall_3d": 182,
        "rainfall_7d": 270,
        "soil_moisture": 0.82,
        "description": "High rainfall hill district with recurrent catastrophic mudslides along NH-37 (Imphal-Jiribam).",
        "evacuation_center": "Tamenglong Higher Secondary Safe Ground",
        "shelter_distance": "0.8 km away",
        "helpline": "03877-222234",
        "sensors_count": 5,
        "population_at_risk": 1450,
    },
    {
        "id": "senapati",
        "name": "Senapati",
        "state": "Manipur",
        "district": "Senapati",
        "coordinates": [25.2700, 94.0200],
        "elevation_m": 1060,
        "slope_degrees": 35,
        "aspect_degrees": 135,
        "rainfall_24h": 70,
        "rainfall_3d": 138,
        "rainfall_7d": 205,
        "soil_moisture": 0.69,
        "description": "NH-2 Lifeline corridor crossing fractured flysch formations with roadblock slips.",
        "evacuation_center": "Senapati Mini Stadium & Town Hall",
        "shelter_distance": "1.1 km away",
        "helpline": "03871-222245",
        "sensors_count": 4,
        "population_at_risk": 880,
    },
    {
        "id": "ukhrul",
        "name": "Ukhrul",
        "state": "Manipur",
        "district": "Ukhrul",
        "coordinates": [25.1200, 94.3600],
        "elevation_m": 1662,
        "slope_degrees": 36,
        "aspect_degrees": 150,
        "rainfall_24h": 75,
        "rainfall_3d": 148,
        "rainfall_7d": 220,
        "soil_moisture": 0.73,
        "description": "Sirohi hill slopes with high moisture retention and weathered fault zones.",
        "evacuation_center": "Ukhrul Higher Secondary Campus Hall",
        "shelter_distance": "0.9 km away",
        "helpline": "03876-222256",
        "sensors_count": 4,
        "population_at_risk": 740,
    },

    # --- MIZORAM ---
    {
        "id": "aizawl",
        "name": "Aizawl",
        "state": "Mizoram",
        "district": "Aizawl",
        "coordinates": [23.7271, 92.7176],
        "elevation_m": 1132,
        "slope_degrees": 38,
        "aspect_degrees": 160,
        "rainfall_24h": 84,
        "rainfall_3d": 172,
        "rainfall_7d": 258,
        "soil_moisture": 0.80,
        "description": "Narrow north-south ridge with high building loads and frequent subsidence at Ramhlun & Laipuitlang.",
        "evacuation_center": "Hawla Indoor Stadium Safe Assembly Area",
        "shelter_distance": "1.4 km away",
        "helpline": "0389-2342520 (State Disaster Mgt Centre)",
        "sensors_count": 8,
        "population_at_risk": 2800,
    },
    {
        "id": "kolasib",
        "name": "Kolasib",
        "state": "Mizoram",
        "district": "Kolasib",
        "coordinates": [24.2200, 92.6800],
        "elevation_m": 590,
        "slope_degrees": 35,
        "aspect_degrees": 130,
        "rainfall_24h": 74,
        "rainfall_3d": 146,
        "rainfall_7d": 218,
        "soil_moisture": 0.72,
        "description": "NH-306 Lifeline highway with recurring monsoon mud-slips and valley subsidence.",
        "evacuation_center": "Kolasib Sub-Divisional Safe Campus Block",
        "shelter_distance": "0.8 km away",
        "helpline": "03837-220023",
        "sensors_count": 4,
        "population_at_risk": 820,
    },

    # --- TRIPURA ---
    {
        "id": "ambassa",
        "name": "Ambassa",
        "state": "Tripura",
        "district": "Dhalai",
        "coordinates": [23.9200, 91.8500],
        "elevation_m": 45,
        "slope_degrees": 26,
        "aspect_degrees": 115,
        "rainfall_24h": 55,
        "rainfall_3d": 110,
        "rainfall_7d": 165,
        "soil_moisture": 0.60,
        "description": "Atharamura & Longthorai hill range crossing with monsoon earth-slips on NH-8.",
        "evacuation_center": "Ambassa Community Welfare Center",
        "shelter_distance": "1.2 km away",
        "helpline": "03826-222234",
        "sensors_count": 3,
        "population_at_risk": 430,
    },
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two GPS coordinates."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def evaluate_location_ml(loc: dict[str, Any]) -> dict[str, Any]:
    """Runs backend ML model on location's geotechnical parameters."""
    feat = {
        "elevation_m": float(loc["elevation_m"]),
        "slope_degrees": float(loc["slope_degrees"]),
        "aspect_degrees": float(loc["aspect_degrees"]),
        "rainfall_1d_before": float(loc["rainfall_24h"]),
        "rainfall_3d_before": float(loc["rainfall_3d"]),
        "rainfall_7d_before": float(loc["rainfall_7d"]),
        "rainfall_14d_before": float(loc["rainfall_7d"] * 1.4),
        "rainfall_30d_before": float(loc["rainfall_7d"] * 2.1),
        "rainfall_7d_max1d": float(loc["rainfall_24h"]),
        "rainfall_3d_over_7d_ratio": round(
            float(loc["rainfall_3d"]) / max(1.0, float(loc["rainfall_7d"])), 3
        ),
        "soil_moisture": float(loc["soil_moisture"]),
        "soil_moisture_available": 1,
    }
    try:
        prepared = FeatureService.prepare(feat)
        result = predictor.predict(prepared)
        return {
            **loc,
            "riskScore": int(round(result.risk_score * 100)),
            "riskTier": result.risk_tier,
            "riskProbability": result.risk_score,
            "alertTriggered": result.alert_triggered,
            "alertMessage": result.alert_message,
        }
    except Exception:
        # Fallback estimation if model weights not loaded
        raw = min(
            0.98,
            max(
                0.15,
                (loc["rainfall_24h"] * 1.5 + loc["rainfall_3d"] * 0.5) / 220
                + (loc["slope_degrees"] / 60) * 0.35
                + loc["soil_moisture"] * 0.25,
            ),
        )
        tier = "CRITICAL" if raw >= 0.85 else "HIGH" if raw >= 0.60 else "MEDIUM" if raw >= 0.30 else "LOW"
        return {
            **loc,
            "riskScore": int(round(raw * 100)),
            "riskTier": tier,
            "riskProbability": round(raw, 3),
            "alertTriggered": tier == "CRITICAL",
            "alertMessage": f"{tier} Landslide Threat Active",
        }


@router.get("")
def get_locations(
    state: str | None = Query(default=None),
    q: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    """Returns Northeast India towns and cities evaluated dynamically by the ML model."""
    results = []
    query_str = (q or "").strip().lower()

    for loc in NORTHEAST_LOCATIONS_DB:
        if state and state != "ALL" and loc["state"].lower() != state.lower():
            continue
        if query_str:
            name_match = query_str in loc["name"].lower()
            dist_match = query_str in loc["district"].lower()
            state_match = query_str in loc["state"].lower()
            if not (name_match or dist_match or state_match):
                continue
        results.append(evaluate_location_ml(loc))

    return results


@router.get("/nearest")
def get_nearest_location(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
) -> dict[str, Any]:
    """Finds the nearest Northeast town to the browser's live GPS coordinates."""
    closest_loc = None
    min_distance = float("inf")

    for loc in NORTHEAST_LOCATIONS_DB:
        d = haversine_distance(lat, lng, loc["coordinates"][0], loc["coordinates"][1])
        if d < min_distance:
            min_distance = d
            closest_loc = loc

    if not closest_loc:
        closest_loc = NORTHEAST_LOCATIONS_DB[0]

    evaluated = evaluate_location_ml(closest_loc)
    evaluated["distanceKm"] = round(min_distance, 1)
    evaluated["liveCoordinates"] = [lat, lng]
    return evaluated


@router.get("/{location_id}")
def get_location_by_id(location_id: str) -> dict[str, Any]:
    for loc in NORTHEAST_LOCATIONS_DB:
        if loc["id"].lower() == location_id.lower():
            return evaluate_location_ml(loc)
    raise HTTPException(status_code=404, detail="Location not found")


@router.get("/{location_id}/dashboard")
def get_location_dashboard(location_id: str) -> dict[str, Any]:
    """Generates complete dynamic LEWS dashboard telemetry for this location."""
    loc = None
    for item in NORTHEAST_LOCATIONS_DB:
        if item["id"].lower() == location_id.lower():
            loc = item
            break
    if not loc:
        loc = NORTHEAST_LOCATIONS_DB[0]

    evaluated = evaluate_location_ml(loc)
    lat, lng = loc["coordinates"]
    risk_score = evaluated["riskScore"]
    risk_tier = evaluated["riskTier"]

    mapped_level = (
        "CRITICAL"
        if risk_tier == "CRITICAL"
        else "HIGH"
        if risk_tier == "HIGH"
        else "WATCH"
        if risk_tier == "MEDIUM"
        else "SAFE"
    )

    # Dynamic risk zones around location
    risk_zones = [
        {
            "id": f"zone-{loc['id']}-01",
            "name": f"{loc['name']} Sector 1 (Ridge Crest)",
            "sectorCode": f"{loc['name'][:3].upper()}-SEC-01",
            "riskScore": risk_score,
            "riskLevel": mapped_level,
            "rainfall24h": loc["rainfall_24h"],
            "soilMoisture": int(round(loc["soil_moisture"] * 100)),
            "slopeAngle": loc["slope_degrees"],
            "historicalEvents": 7,
            "predictionWindow": "2–4 hours" if risk_tier == "CRITICAL" else "4–8 hours",
            "confidence": 92,
            "coordinates": [
                [lat + 0.006, lng - 0.007],
                [lat + 0.013, lng + 0.003],
                [lat + 0.004, lng + 0.012],
                [lat - 0.006, lng + 0.002],
                [lat - 0.003, lng - 0.008],
            ],
            "center": [lat + 0.003, lng + 0.001],
            "elevation": loc["elevation_m"],
            "description": loc["description"],
            "sensorsCount": loc["sensors_count"],
            "populationAtRisk": loc["population_at_risk"],
            "nearestFacility": loc["evacuation_center"],
        },
        {
            "id": f"zone-{loc['id']}-02",
            "name": f"{loc['name']} Highway Transit Flank",
            "sectorCode": f"{loc['name'][:3].upper()}-HWY-02",
            "riskScore": max(20, risk_score - 14),
            "riskLevel": "HIGH" if risk_tier == "CRITICAL" else "WATCH",
            "rainfall24h": int(loc["rainfall_24h"] * 0.9),
            "soilMoisture": int(round(loc["soil_moisture"] * 90)),
            "slopeAngle": max(18, loc["slope_degrees"] - 5),
            "historicalEvents": 4,
            "predictionWindow": "6–12 hours",
            "confidence": 88,
            "coordinates": [
                [lat - 0.014, lng - 0.010],
                [lat - 0.007, lng - 0.002],
                [lat - 0.013, lng + 0.008],
                [lat - 0.021, lng - 0.001],
            ],
            "center": [lat - 0.013, lng + 0.001],
            "elevation": int(loc["elevation_m"] * 0.94),
            "description": "Arterial road cut slope with rockfall protection wire and inclinometer sensors.",
            "sensorsCount": 4,
            "populationAtRisk": int(loc["population_at_risk"] * 0.55),
            "nearestFacility": loc["evacuation_center"],
        },
    ]

    return {
        "id": loc["id"],
        "name": f"{loc['name']} ({loc['district']})",
        "state": loc["state"],
        "center": [lat, lng],
        "zoom": 13,
        "currentRisk": risk_score,
        "riskLevel": mapped_level,
        "riskTrend": "+12% over last 24h" if risk_score > 60 else "Stable slope trend",
        "predictionWindow": "Critical alert active" if risk_tier == "CRITICAL" else "Next 6-12 hours",
        "confidence": 91,
        "criticalAlertsCount": 2 if risk_tier in ("CRITICAL", "HIGH") else 0,
        "highRiskZonesCount": len(risk_zones),
        "blockedRoadsCount": 1 if risk_tier == "CRITICAL" else 0,
        "environmental": {
            "rainfall24h": loc["rainfall_24h"],
            "soilMoisture": int(round(loc["soil_moisture"] * 100)),
            "temperature": 15 if loc["elevation_m"] > 2000 else 23,
            "groundMovement": 1.6 if risk_tier == "CRITICAL" else 0.4,
            "humidity": 88,
            "windSpeed": 14,
        },
        "explainability": [
            {
                "factor": "Antecedent Precipitation",
                "percentage": 36,
                "metricValue": f"{loc['rainfall_24h']} mm / 24h",
                "category": "rainfall",
                "impact": "high" if loc["rainfall_24h"] > 60 else "moderate",
            },
            {
                "factor": "Soil Saturation Ratio",
                "percentage": 28,
                "metricValue": f"{int(round(loc['soil_moisture'] * 100))}% saturation",
                "category": "soil",
                "impact": "high" if loc["soil_moisture"] > 0.7 else "moderate",
            },
            {
                "factor": "Slope Incline",
                "percentage": 22,
                "metricValue": f"{loc['slope_degrees']}° critical grade",
                "category": "slope",
                "impact": "high" if loc["slope_degrees"] > 35 else "moderate",
            },
            {
                "factor": "Altitude & Orography",
                "percentage": 14,
                "metricValue": f"{loc['elevation_m']} m above MSL",
                "category": "movement",
                "impact": "moderate",
            },
        ],
        "recommendedActions": [
            {
                "id": f"act-{loc['id']}-1",
                "priority": 1,
                "title": f"Deploy Geotechnical Team to {loc['name']} Sector 1",
                "location": f"{loc['name']} Ridge Crest",
                "status": "Patrol active" if risk_tier in ("CRITICAL", "HIGH") else "Standby",
                "actionType": "INSPECT",
                "description": "Inspect tension cracks and verify inclinometer readings near residential slopes.",
                "assignedAgency": "PWD Geotechnical & NDRF Disaster Unit",
                "targetCompletion": "Within 2 hrs",
            },
            {
                "id": f"act-{loc['id']}-2",
                "priority": 2,
                "title": f"Review Transit Clearance on Arterial Highway",
                "location": f"{loc['name']} Transit Arterial",
                "status": "Patrol active",
                "actionType": "MONITOR",
                "description": "Clear loose scree and keep drone surveillance active along high-cut road spurs.",
                "assignedAgency": "Border Roads Organisation (BRO) / NHIDCL",
                "targetCompletion": "Continuous 24h Watch",
            },
            {
                "id": f"act-{loc['id']}-3",
                "priority": 3,
                "title": f"Stage Supplies at {loc['evacuation_center']}",
                "location": loc["evacuation_center"],
                "status": "Standby",
                "actionType": "PREPARE_EVACUATION",
                "description": "Verify emergency diesel generators, potable water rations, and medical kits.",
                "assignedAgency": "District Disaster Management Authority",
                "targetCompletion": "Immediate",
            },
        ],
        "riskZones": risk_zones,
        "alerts": [
            {
                "id": f"alt-{loc['id']}-01",
                "title": f"{risk_tier} Landslide Threat in {loc['name']}",
                "location": f"{loc['name']} Sector 1",
                "riskScore": risk_score,
                "severity": mapped_level,
                "timestamp": "Just now",
                "timeAgo": "10m ago",
                "summary": f"{loc['name']} is experiencing {loc['rainfall_24h']}mm 24h rainfall. Evacuate unstable slopes immediately.",
                "affectedRoads": ["Main Hill Road", "Highway Spur Km 42"],
                "recommendedAction": "Immediate evacuation of downhill settlements",
                "status": "ACTIVE",
            }
        ],
        "roads": [
            {
                "id": f"rd-{loc['id']}-01",
                "name": f"{loc['name']} Primary Highway Corridor",
                "highwayCode": "NH-13 / Inter-District Pass",
                "status": "RESTRICTED" if risk_tier == "CRITICAL" else "CLEAR",
                "blockageReason": "Monsoon debris and minor rockfall on road shoulder.",
                "coordinates": [
                    [lat - 0.015, lng - 0.018],
                    [lat - 0.008, lng - 0.005],
                    [lat + 0.005, lng + 0.008],
                    [lat + 0.015, lng + 0.020],
                ],
                "affectedLengthKm": 3.8,
                "clearanceEstimate": "2-4 hours if cleared by BRO",
                "bypassAvailable": True,
                "bypassRouteName": "Valley Link Spur Road",
                "lastUpdated": "15m ago",
            }
        ],
        "sensors": [
            {
                "id": f"sens-{loc['id']}-rg1",
                "name": f"RG-{loc['name'][:3].upper()}-01",
                "type": "RAIN_GAUGE",
                "coordinates": [lat + 0.004, lng + 0.002],
                "value": f"{loc['rainfall_24h']} mm / 24h",
                "status": "OPTIMAL",
                "battery": 98,
                "lastPing": "1m ago",
            },
            {
                "id": f"sens-{loc['id']}-sm1",
                "name": f"SM-{loc['name'][:3].upper()}-02",
                "type": "SOIL_MOISTURE",
                "coordinates": [lat + 0.002, lng - 0.004],
                "value": f"{int(round(loc['soil_moisture'] * 100))}% saturation",
                "status": "WARNING" if loc["soil_moisture"] > 0.75 else "OPTIMAL",
                "battery": 94,
                "lastPing": "3m ago",
            },
        ],
        "facilities": [
            {
                "id": f"fac-{loc['id']}-01",
                "name": loc["evacuation_center"],
                "type": "EVACUATION_CENTER",
                "coordinates": [lat - 0.005, lng + 0.005],
                "capacity": 450,
                "occupancy": 65,
                "contact": loc["helpline"],
            }
        ],
        "fieldReports": [],
        "trend24h": [
            {"time": "14:00 (Y)", "risk": max(15, risk_score - 20), "rainfall": int(loc["rainfall_24h"] * 0.4), "threshold": 70},
            {"time": "20:00", "risk": max(20, risk_score - 15), "rainfall": int(loc["rainfall_24h"] * 0.6), "threshold": 70},
            {"time": "02:00", "risk": max(25, risk_score - 8), "rainfall": int(loc["rainfall_24h"] * 0.8), "threshold": 70},
            {"time": "08:00", "risk": risk_score, "rainfall": loc["rainfall_24h"], "threshold": 70},
            {"time": "14:00 (Now)", "risk": risk_score, "rainfall": loc["rainfall_24h"], "threshold": 70},
        ],
    }
