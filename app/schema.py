"""
GraphQL schema exposing the migrated Digital Twin functionality.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import strawberry
from strawberry.scalars import JSON

from app.services.battery_service import (
    create_battery,
    delete_battery,
    get_battery,
    list_batteries,
    update_battery,
)
from app.services.appliance_service import (
    attach_measurement,
    clear_measurement,
    create_appliance,
    delete_appliance,
    get_appliance,
    list_appliances,
    update_appliance,
)
from app.services.inverter_service import (
    create_inverter,
    delete_inverter,
    get_inverter,
    list_inverters,
    update_inverter,
)
from app.services.panel_service import (
    create_panel,
    delete_panel,
    get_panel,
    list_panels,
    update_panel,
)
from app.services.prediction_service import get_predictions_bundle
from app.services.solar_service import get_solar_snapshot
from app.services.system_config import get_system_config
from app.services.user_service import (
    authenticate_user,
    authenticate_or_provision_ldap,
    change_password,
    register_user,
    list_users,
    is_admin,
    delete_user,
)
from app.services.session_service import (
    create_session,
    list_active_sessions,
    revoke_session,
    revoke_sessions_by_email,
)
from app.services.ldap_config_service import (
    get_ldap_config,
    save_ldap_config,
    is_ldap_enabled,
    test_ldap_connection,
)
from app.services.weather_service import get_weather_with_fallback
from app.services.weather_source_service import (
    delete_weather_source,
    get_active_weather_source,
    list_weather_sources,
    save_weather_source,
    set_active_weather_source,
    test_weather_source,
)
from app.services.location_config_service import (
    get_location_config,
    save_location_config,
)
from app.services.shadow_profile_service import (
    get_shadow_profile,
    save_shadow_profile,
)
from app.services.ml_prediction_service import (
    predict_solar_production,
    predict_next_hours,
    predict_for_date_range,
)
from app.services.ml_model_service import ml_model_service
from app.services.battery_discharge_service import calculate_battery_discharge_time
from app.services.invitation_service import create_invitation_code, list_invitation_codes
from app.database import get_database
import strawberry.types
from app.auth import create_token, require_admin, require_auth
from app.services.lectura_service import (
    get_readings,
    get_daily_summaries,
    save_reading,
    seed_historical_data,
)
from app.services.panel_cleanliness_service import get_latest_cleanliness
from app.services.medicion_service import (
    upload_batch as upload_appliance_batch,
    delete_batch as delete_appliance_batch,
    list_batches as list_appliance_batches,
    get_daily_report,
    get_appliance_readings,
    preview_batch,
)


# ============================================================================
# Types
# ============================================================================


@strawberry.type
class SolarPoint:
    timestamp: str
    production: float
    consumption: float
    batteryLevel: float
    gridExport: float
    gridImport: float
    efficiency: float
    batteryDelta: Optional[float]


@strawberry.type
class BatteryStatusType:
    chargeLevel: float
    capacity: float
    current: float
    autonomyHours: float
    charging: bool
    powerFlow: float
    projectedMinLevel: Optional[float]
    projectedMaxLevel: Optional[float]
    note: Optional[str]


@strawberry.type
class SystemMetricsType:
    currentProduction: float
    currentConsumption: float
    energyBalance: float
    systemEfficiency: float
    dailyProduction: float
    dailyConsumption: float
    co2Avoided: float


@strawberry.type
class EnergyFlowType:
    solarToBattery: float
    solarToLoad: float
    solarToGrid: float
    batteryToLoad: float
    gridToLoad: float


@strawberry.type
class WeatherForecastDay:
    date: str
    dayOfWeek: str
    maxTemp: float
    minTemp: float
    solarRadiation: float
    cloudCover: float
    predictedProduction: float
    condition: str


@strawberry.type
class WeatherDataType:
    temperature: float
    solarRadiation: float
    cloudCover: float
    humidity: float
    windSpeed: float
    forecast: List[WeatherForecastDay]
    provider: Optional[str]
    locationName: Optional[str]
    lastUpdated: Optional[str]
    description: Optional[str]
    weatherCode: Optional[int]
    sourceError: Optional[str]
    isMock: bool = False


@strawberry.type
class LocationConfigType:
    lat: float
    lon: float
    name: str


@strawberry.type
class LocationConfigExtType:
    lat: float
    lon: float
    name: str
    updatedAt: Optional[str]


@strawberry.type
class ShadowSlotType:
    hour: int
    shadow_pct: float = strawberry.field(name="shadowPct")
    prod_override: Optional[float] = strawberry.field(name="prodOverride")


@strawberry.type
class ShadowProfileType:
    slots: List[ShadowSlotType]
    avg_shadow: float = strawberry.field(name="avgShadow")
    avg_prod: float = strawberry.field(name="avgProd")
    updated_at: Optional[str] = strawberry.field(name="updatedAt")


@strawberry.input
class ShadowSlotInput:
    hour: int
    shadow_pct: float = strawberry.field(name="shadowPct")
    prod_override: Optional[float] = strawberry.field(name="prodOverride", default=None)


@strawberry.type
class PanelConfigSpec:
    id_: Optional[str] = strawberry.field(name="_id")
    manufacturer: Optional[str]
    model: Optional[str]
    ratedPowerKw: Optional[float]
    quantity: Optional[int]
    tiltDegrees: Optional[float]
    orientation: Optional[str]
    efficiencyPercent: Optional[float] = None
    areaM2: Optional[float] = None
    createdAt: Optional[str]
    updatedAt: Optional[str]


@strawberry.type
class SolarConfigType:
    capacityKw: float
    panelRatedKw: Optional[float]
    panelCount: int
    strings: Optional[int]
    panelEfficiencyPercent: Optional[float]
    panelAreaM2: Optional[float]
    spec: Optional[PanelConfigSpec]


@strawberry.type
class BatteryConfigSpec:
    id_: Optional[str] = strawberry.field(name="_id")
    manufacturer: Optional[str]
    model: Optional[str]
    capacityKwh: Optional[float]
    quantity: Optional[int]
    maxDepthOfDischargePercent: Optional[float] = None
    chargeRateKw: Optional[float] = None
    dischargeRateKw: Optional[float] = None
    efficiencyPercent: Optional[float] = None
    createdAt: Optional[str]
    updatedAt: Optional[str]


@strawberry.type
class BatteryConfigType:
    capacityKwh: float
    moduleCapacityKwh: Optional[float]
    moduleCount: Optional[int]
    maxDepthOfDischargePercent: Optional[float]
    chargeRateKw: Optional[float]
    dischargeRateKw: Optional[float]
    efficiencyPercent: Optional[float]
    spec: Optional[BatteryConfigSpec]


@strawberry.type
class SystemConfigType:
    location: LocationConfigType
    solar: SolarConfigType
    battery: BatteryConfigType


@strawberry.type
class SolarSnapshot:
    current: SolarPoint
    historical: List[SolarPoint]
    battery: BatteryStatusType
    metrics: SystemMetricsType
    energyFlow: EnergyFlowType
    weather: WeatherDataType
    config: SystemConfigType
    timestamp: str
    mode: str


@strawberry.type
class PredictionType:
    timestamp: str
    hour: int
    expectedProduction: float
    expectedConsumption: float
    confidence: float


@strawberry.type
class AlertType:
    id: str
    type: str
    title: str
    message: str
    timestamp: str


@strawberry.type
class PredictionsPayload:
    predictions: List[PredictionType]
    alerts: List[AlertType]
    recommendations: List[str]
    battery: BatteryStatusType
    timeline: List[SolarPoint]
    weather: WeatherDataType
    timestamp: str
    config: SystemConfigType


@strawberry.type
class PanelType:
    id_: str = strawberry.field(name="_id")
    manufacturer: str
    model: Optional[str]
    ratedPowerKw: float
    quantity: int
    tiltDegrees: Optional[float]
    orientation: Optional[str]
    efficiencyPercent: Optional[float] = None
    areaM2: Optional[float] = None
    createdAt: Optional[str]
    updatedAt: Optional[str]


@strawberry.type
class BatteryType:
    id_: str = strawberry.field(name="_id")
    manufacturer: str
    model: Optional[str]
    capacityKwh: float
    quantity: int
    maxDepthOfDischargePercent: Optional[float] = None
    chargeRateKw: Optional[float] = None
    dischargeRateKw: Optional[float] = None
    efficiencyPercent: Optional[float] = None
    createdAt: Optional[str]
    updatedAt: Optional[str]


@strawberry.type
class ApplianceModeType:
    name: str
    averagePowerW: float
    maxPowerW: Optional[float]


@strawberry.type
class ApplianceMeasurementMetaType:
    samples: int
    firstDate: Optional[str]
    lastDate: Optional[str]
    avgKw: float
    minKw: float
    maxKw: float
    stdKw: float
    hoursCovered: int


@strawberry.type
class ApplianceType:
    id_: str = strawberry.field(name="_id")
    name: str
    category: Optional[str]
    averagePowerW: float
    maxPowerW: float
    measuredPowerW: Optional[float]
    quantity: int
    activeHours: Optional[float]
    selectedModeIndex: Optional[int]
    modes: List[ApplianceModeType]
    alwaysOn: bool = True
    useMeasurements: bool = False
    activeHourMask: List[int] = strawberry.field(default_factory=list)
    uncoveredHoursFill: str = "mean"
    hourlyProfileKw: List[float] = strawberry.field(default_factory=list)
    measurementMeta: Optional[ApplianceMeasurementMetaType] = None
    createdAt: Optional[str]
    updatedAt: Optional[str]


@strawberry.type
class HourlyForecastPoint:
    datetime: str
    consumptionKw: float


@strawberry.type
class ApplianceForecastSummary:
    totalConsumptionKw: float
    appliancesWithProfile: int
    appliancesAlwaysOn: int
    points: List[HourlyForecastPoint]


@strawberry.type
class InverterType:
    id_: str = strawberry.field(name="_id")
    manufacturer: str
    model: Optional[str]
    ratedPowerKw: float
    quantity: int
    efficiencyPercent: Optional[float]
    createdAt: Optional[str]
    updatedAt: Optional[str]


@strawberry.type
class UserType:
    id_: str = strawberry.field(name="_id")
    email: str
    name: Optional[str]
    role: str
    createdAt: Optional[str]
    updatedAt: Optional[str]


@strawberry.type
class SessionType:
    id_: str = strawberry.field(name="_id")
    email: str
    ip: str
    userAgent: str
    deviceType: str
    createdAt: Optional[str]
    expiresAt: Optional[str]


@strawberry.type
class AuthPayloadType:
    user: UserType
    token: str


@strawberry.type
class HistoricalReadingType:
    id_: str = strawberry.field(name="_id")
    timestamp: str
    productionKw: float


@strawberry.type
class PanelCleanlinessType:
    """Última lectura de limpieza de paneles (clasificador de imágenes)."""
    id_: str = strawberry.field(name="_id")
    timestamp: str
    clasificacion: str
    porcentajeLimpio: float
    porcentajeSucio: float
    source: str


@strawberry.type
class DailySummaryType:
    date: str
    totalProductionKwh: float
    maxProductionKw: float
    readingCount: int


@strawberry.type
class ApplianceBatchType:
    batchId: str
    applianceId: str
    applianceName: str
    filename: str
    uploadedAt: str
    startDate: str
    endDate: str
    samples: int
    kwhDayEstimatedThis: float
    kwhDayEstimatedOthers: float


@strawberry.type
class BatchPreviewType:
    samples: int
    startDate: Optional[str]
    endDate: Optional[str]


@strawberry.type
class ApplianceReadingPointType:
    timestamp: str
    powerKw: float


@strawberry.type
class DailyReportApplianceType:
    applianceId: str
    name: str
    mode: str  # "medido" | "estimado"
    kwhDay: float
    kwhDayEstimated: Optional[float]
    errorPercent: Optional[float]
    readingCount: int


@strawberry.type
class DailyReportType:
    date: str
    productionKwh: Optional[float]
    measuredConsumptionKwh: float
    estimatedConsumptionKwh: float
    totalConsumptionKwh: float
    hasRealData: bool
    appliances: List[DailyReportApplianceType]


@strawberry.type
class MLWeatherFeaturesType:
    temperature_2m: float
    relative_humidity_2m: float
    wind_speed_10m: float
    cloud_cover: float
    shortwave_radiation: float


@strawberry.type
class MLPredictionType:
    datetime: str
    production_kw: float
    weather: MLWeatherFeaturesType
    # Fuente de datos meteorológicos usada por el modelo. El ML está entrenado
    # contra Open-Meteo, así que este campo siempre vale "Open-Meteo".
    weather_source: str = strawberry.field(name="weatherSource", default="Open-Meteo")
    # Si la fuente configurada por el usuario es distinta, aquí se devuelve un
    # aviso para que el frontend lo muestre y no se vea como inconsistencia.
    weather_source_warning: Optional[str] = strawberry.field(
        name="weatherSourceWarning", default=None
    )


@strawberry.type
class MLModelInfoType:
    loaded: bool
    model_name: Optional[str]
    test_rmse: Optional[float]
    test_r2: Optional[float]
    test_mae: Optional[float]
    features: List[str]
    training_date: Optional[str]
    requires_scaling: Optional[bool]
    reference_capacity_kw: Optional[float]
    message: Optional[str]


@strawberry.type
class MonthlyProductionType:
    """Producción de un mes del año, con su origen.

    ``source`` indica de dónde sale la cifra:
      * ``historico``   — meses ya pasados (clima real de Open-Meteo).
      * ``prediccion``  — meses futuros (climatología: mismo mes del año anterior).
      * ``mixto``       — mes en curso (parte real + parte estimada).
    """
    month: int          # 1..12
    monthName: str
    productionKwh: float
    source: str


@strawberry.type
class BatteryDischargeEstimateType:
    minutesToEmpty: Optional[int]
    startHour: int
    batteryCapacityKwh: float


@strawberry.type
class InvitationCodeType:
    id_: str = strawberry.field(name="_id")
    code: str
    role: str
    isUsed: bool
    createdBy: Optional[str]
    usedBy: Optional[str]
    createdAt: Optional[str]
    updatedAt: Optional[str]


@strawberry.type
class LdapConfigType:
    enabled: bool
    serverUrl: str
    baseDn: str
    bindDn: str
    userSearchFilter: str
    emailAttr: str
    nameAttr: str
    useTls: bool
    connectTimeout: int
    # La contraseña de bind nunca se devuelve; solo se indica si hay una guardada.
    hasBindPassword: bool
    updatedAt: Optional[str]


@strawberry.type
class LdapTestResultType:
    success: bool
    message: str
    sampleUser: Optional[str]


@strawberry.type
class WeatherSourceType:
    id_: str = strawberry.field(name="_id")
    name: str
    baseUrl: Optional[str]
    authType: str
    authHeaderName: Optional[str]
    authQueryName: Optional[str]
    authValue: Optional[str]
    queryParams: JSON
    fieldMapping: JSON
    locationName: Optional[str]
    enabled: bool
    isActive: bool
    createdAt: Optional[str]
    updatedAt: Optional[str]


@strawberry.type
class WeatherFieldCandidateType:
    path: str
    valueType: str
    sampleValue: str


@strawberry.type
class WeatherSourceTestResultType:
    success: bool
    message: str
    fields: List[WeatherFieldCandidateType]
    rawJson: str


# ============================================================================
# Helpers
# ============================================================================


def _map_solar_point(item: dict) -> SolarPoint:
    return SolarPoint(**item)


def _map_weather(data: dict) -> WeatherDataType:
    return WeatherDataType(
        temperature=data["temperature"],
        solarRadiation=data["solarRadiation"],
        cloudCover=data["cloudCover"],
        humidity=data["humidity"],
        windSpeed=data["windSpeed"],
        forecast=[WeatherForecastDay(**day) for day in data.get("forecast", [])],
        provider=data.get("provider"),
        locationName=data.get("locationName"),
        lastUpdated=data.get("lastUpdated"),
        description=data.get("description"),
        weatherCode=data.get("weatherCode"),
        sourceError=data.get("sourceError"),
        isMock=bool(data.get("isMock", False)),
    )


def _map_battery_status(data: dict) -> BatteryStatusType:
    return BatteryStatusType(**data)


def _map_metrics(data: dict) -> SystemMetricsType:
    return SystemMetricsType(**data)


def _map_energy_flow(data: dict) -> EnergyFlowType:
    return EnergyFlowType(**data)


def _map_panel(data: dict) -> PanelType:
    # Rename _id to id_ for Strawberry field mapping
    data_copy = {**data}
    if "_id" in data_copy:
        data_copy["id_"] = data_copy.pop("_id")
    return PanelType(**data_copy)


def _map_battery(data: dict) -> BatteryType:
    # Rename _id to id_ for Strawberry field mapping
    data_copy = {**data}
    if "_id" in data_copy:
        data_copy["id_"] = data_copy.pop("_id")
    return BatteryType(**data_copy)


def _map_appliance_mode(data: dict) -> ApplianceModeType:
    return ApplianceModeType(
        name=data.get("name", ""),
        averagePowerW=data.get("averagePowerW", 0),
        maxPowerW=data.get("maxPowerW"),
    )


def _map_appliance(data: dict) -> ApplianceType:
    data_copy = {**data}
    if "_id" in data_copy:
        data_copy["id_"] = data_copy.pop("_id")
    data_copy["modes"] = [_map_appliance_mode(mode) for mode in data.get("modes", [])]
    data_copy["alwaysOn"] = True if data.get("alwaysOn") is None else bool(data.get("alwaysOn"))
    data_copy["useMeasurements"] = bool(data.get("useMeasurements", False))
    data_copy["hourlyProfileKw"] = list(data.get("hourlyProfileKw") or [])
    meta = data.get("measurementMeta")
    data_copy["measurementMeta"] = (
        ApplianceMeasurementMetaType(
            samples=int(meta.get("samples", 0)),
            firstDate=meta.get("firstDate"),
            lastDate=meta.get("lastDate"),
            avgKw=float(meta.get("avgKw", 0.0)),
            minKw=float(meta.get("minKw", 0.0)),
            maxKw=float(meta.get("maxKw", 0.0)),
            stdKw=float(meta.get("stdKw", 0.0)),
            hoursCovered=int(meta.get("hoursCovered", 0)),
        )
        if isinstance(meta, dict)
        else None
    )
    return ApplianceType(**data_copy)


def _map_inverter(data: dict) -> InverterType:
    # Rename _id to id_ for Strawberry field mapping
    data_copy = {**data}
    if "_id" in data_copy:
        data_copy["id_"] = data_copy.pop("_id")
    return InverterType(**data_copy)


def _map_panel_spec(data: dict) -> PanelConfigSpec:
    # Rename _id to id_ for Strawberry field mapping
    data_copy = {**data}
    if "_id" in data_copy:
        data_copy["id_"] = data_copy.pop("_id")
    return PanelConfigSpec(**data_copy)


def _map_battery_spec(data: dict) -> BatteryConfigSpec:
    # Rename _id to id_ for Strawberry field mapping
    data_copy = {**data}
    if "_id" in data_copy:
        data_copy["id_"] = data_copy.pop("_id")
    return BatteryConfigSpec(**data_copy)


def _map_system_config(config: dict) -> SystemConfigType:
    loc = config["location"]
    location = LocationConfigType(lat=loc["lat"], lon=loc["lon"], name=loc["name"])
    solar_spec = _map_panel_spec(config["solar"]["spec"]) if config["solar"].get("spec") else None
    battery_spec = _map_battery_spec(config["battery"]["spec"]) if config["battery"].get("spec") else None
    solar = SolarConfigType(
        capacityKw=config["solar"]["capacityKw"],
        panelRatedKw=config["solar"].get("panelRatedKw"),
        panelCount=config["solar"].get("panelCount") or 0,
        strings=config["solar"].get("strings"),
        panelEfficiencyPercent=config["solar"].get("panelEfficiencyPercent"),
        panelAreaM2=config["solar"].get("panelAreaM2"),
        spec=solar_spec,
    )
    battery = BatteryConfigType(
        capacityKwh=config["battery"]["capacityKwh"],
        moduleCapacityKwh=config["battery"].get("moduleCapacityKwh"),
        moduleCount=config["battery"].get("moduleCount"),
        maxDepthOfDischargePercent=config["battery"].get("maxDepthOfDischargePercent"),
        chargeRateKw=config["battery"].get("chargeRateKw"),
        dischargeRateKw=config["battery"].get("dischargeRateKw"),
        efficiencyPercent=config["battery"].get("efficiencyPercent"),
        spec=battery_spec,
    )
    return SystemConfigType(location=location, solar=solar, battery=battery)


def _map_user(data: dict) -> UserType:
    # Rename _id to id_ for Strawberry field mapping
    data_copy = {**data}
    if "_id" in data_copy:
        data_copy["id_"] = data_copy.pop("_id")
    return UserType(**data_copy)


def _map_ldap_config(cfg: dict) -> "LdapConfigType":
    return LdapConfigType(
        enabled=bool(cfg.get("enabled")),
        serverUrl=cfg.get("serverUrl") or "",
        baseDn=cfg.get("baseDn") or "",
        bindDn=cfg.get("bindDn") or "",
        userSearchFilter=cfg.get("userSearchFilter") or "",
        emailAttr=cfg.get("emailAttr") or "",
        nameAttr=cfg.get("nameAttr") or "",
        useTls=bool(cfg.get("useTls")),
        connectTimeout=int(cfg.get("connectTimeout") or 5),
        hasBindPassword=bool(cfg.get("hasBindPassword")),
        updatedAt=cfg.get("updatedAt"),
    )


def _map_weather_source(data: dict) -> WeatherSourceType:
    data_copy = {**data}
    if "_id" in data_copy:
        data_copy["id_"] = data_copy.pop("_id")
    data_copy["queryParams"] = data_copy.get("queryParams") or {}
    data_copy["fieldMapping"] = data_copy.get("fieldMapping") or {}
    return WeatherSourceType(**data_copy)


def _get_real_capacity_kw_from_config(config: Optional[dict]) -> Optional[float]:
    if not config:
        return None
    solar_cfg = config.get("solar") or {}
    capacity = solar_cfg.get("capacityKw")
    try:
        return float(capacity) if capacity is not None else None
    except (TypeError, ValueError):
        return None


def _scale_ml_predictions(
    predictions: List[Dict[str, Any]],
    capacity_kw: Optional[float] = None,
) -> List[Dict[str, Any]]:
    reference_capacity_kw = ml_model_service.get_reference_capacity_kw()
    if not reference_capacity_kw or reference_capacity_kw <= 0:
        return predictions

    target_capacity_kw = capacity_kw
    if target_capacity_kw is None:
        try:
            config = get_system_config()
        except Exception:
            config = None
        target_capacity_kw = _get_real_capacity_kw_from_config(config)

    if not target_capacity_kw or target_capacity_kw <= 0:
        return [{**pred, "production_kw": 0.0} for pred in predictions]

    scale_factor = target_capacity_kw / reference_capacity_kw
    scaled_predictions: List[Dict[str, Any]] = []
    for pred in predictions:
        scaled_predictions.append({
            **pred,
            "production_kw": round(float(pred["production_kw"]) * scale_factor, 2),
        })
    return scaled_predictions


def _to_ml_prediction_type(pred: Dict[str, Any]) -> "MLPredictionType":
    return MLPredictionType(
        datetime=pred["datetime"],
        production_kw=pred["production_kw"],
        weather=MLWeatherFeaturesType(**pred["weather"]),
        weather_source=pred.get("weather_source", "Open-Meteo"),
        weather_source_warning=pred.get("weather_source_warning"),
    )


# ============================================================================
# Queries
# ============================================================================


@strawberry.type
class Query:
    @strawberry.field
    async def solar(self) -> SolarSnapshot:
        data = await get_solar_snapshot()
        return SolarSnapshot(
            current=_map_solar_point(data["current"]),
            historical=[_map_solar_point(item) for item in data["historical"]],
            battery=_map_battery_status(data["battery"]),
            metrics=_map_metrics(data["metrics"]),
            energyFlow=_map_energy_flow(data["energyFlow"]),
            weather=_map_weather(data["weather"]),
            config=_map_system_config(data["config"]),
            timestamp=data["timestamp"],
            mode=data["mode"],
        )

    @strawberry.field
    async def weather(self) -> WeatherDataType:
        config = get_system_config()
        weather = await get_weather_with_fallback(
            config["location"]["lat"],
            config["location"]["lon"],
            config["solar"]["capacityKw"],
            config["location"]["name"],
        )
        return _map_weather(weather)

    @strawberry.field
    async def predictions(self, hours: int = 24) -> PredictionsPayload:
        data = await get_predictions_bundle(hours=hours)
        return PredictionsPayload(
            predictions=[PredictionType(**prediction) for prediction in data["predictions"]],
            alerts=[AlertType(**alert) for alert in data["alerts"]],
            recommendations=data["recommendations"],
            battery=_map_battery_status(data["battery"]),
            timeline=[_map_solar_point(item) for item in data["timeline"]],
            weather=_map_weather(data["weather"]),
            timestamp=data["timestamp"],
            config=_map_system_config(data["config"]),
        )

    @strawberry.field
    def panels(self) -> List[PanelType]:
        return [_map_panel(panel) for panel in list_panels()]

    @strawberry.field
    def panel(self, id: str) -> Optional[PanelType]:
        panel = get_panel(id)
        return _map_panel(panel) if panel else None

    @strawberry.field
    def batteries(self) -> List[BatteryType]:
        return [_map_battery(battery) for battery in list_batteries()]

    @strawberry.field
    def battery(self, id: str) -> Optional[BatteryType]:
        battery = get_battery(id)
        return _map_battery(battery) if battery else None

    @strawberry.field
    def appliances(self) -> List[ApplianceType]:
        return [_map_appliance(appliance) for appliance in list_appliances()]

    @strawberry.field
    def appliance(self, id: str) -> Optional[ApplianceType]:
        appliance = get_appliance(id)
        return _map_appliance(appliance) if appliance else None

    @strawberry.field
    def appliancesConsumptionForecast(
        self,
        hours: int = 24,
        start: Optional[str] = None,
    ) -> ApplianceForecastSummary:
        """
        Sum the forecasted consumption (kW) of every appliance flagged as
        alwaysOn for the next `hours` starting at `start` (ISO datetime,
        defaults to the next full hour). Appliances with an uploaded
        measurement profile use their (weekday x hour) average; the rest
        contribute averagePowerW * quantity converted to kW.
        """
        # ========================================================================
        # MOTOR DEL ENSAMBLE DE CONSUMO (añadido para la integración)
        # ========================================================================
        try:
            from datetime import datetime as _dt_ens, timedelta as _td_ens
            from zoneinfo import ZoneInfo as _ZoneInfo_ens
            from app.services.prediction_service import get_consumption_engine

            engine = get_consumption_engine()

            if start:
                begin_ens = _dt_ens.fromisoformat(start.replace("Z", "+00:00")).replace(tzinfo=None)
            else:
                now_ens = _dt_ens.now(_ZoneInfo_ens("America/Havana")).replace(tzinfo=None)
                begin_ens = now_ens.replace(minute=0, second=0, microsecond=0)

            hours_ens = max(1, min(int(hours), 24 * 14))
            datetimes_ens = [(begin_ens + _td_ens(hours=i)).isoformat() for i in range(hours_ens)]

            predicciones = engine.predict(datetimes_ens, config={})

            points_ens = [
                HourlyForecastPoint(
                    datetime=dt,
                    consumptionKw=round(float(pred), 4),
                )
                for dt, pred in zip(datetimes_ens, predicciones)
            ]
            total_ens = sum(predicciones)

            return ApplianceForecastSummary(
                totalConsumptionKw=round(total_ens, 4),
                appliancesWithProfile=0,
                appliancesAlwaysOn=0,
                points=points_ens,
            )
        except Exception as e:
            print(f"⚠️  Ensamble de consumo falló, usando fallback: {e}")
       # ========================================================================   
     
        from datetime import datetime as _dt, timedelta as _td
        from zoneinfo import ZoneInfo as _ZoneInfo

        from app.services.appliance_measurement_service import forecast_kw

        if start:
            try:
                begin = _dt.fromisoformat(start.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                begin = _dt.now(_ZoneInfo("America/Havana")).replace(tzinfo=None)
        else:
            # Hora local de La Habana (igual que predictions / mlPredictNextHours)
            # para que el consumo se alinee por hora con esas series.
            now = _dt.now(_ZoneInfo("America/Havana")).replace(tzinfo=None)
            begin = now.replace(minute=0, second=0, microsecond=0)

        hours = max(1, min(int(hours), 24 * 14))
        items = list_appliances()

        points: List[HourlyForecastPoint] = []
        always_on_count = 0
        with_profile_count = 0
        for appliance in items:
            if not (True if appliance.get("alwaysOn") is None else bool(appliance.get("alwaysOn"))):
                continue
            always_on_count += 1
            if appliance.get("hourlyProfileKw"):
                with_profile_count += 1

        # Precompute per-appliance shape: profile / scheduled mask / flat 24h.
        prepared = []
        for appliance in items:
            profile = appliance.get("hourlyProfileKw") or []
            if len(profile) == 168:
                prepared.append(("profile", profile, None))
                continue
            avg_w = float(appliance.get("averagePowerW") or 0)
            qty = int(appliance.get("quantity") or 1)
            power_kw = (avg_w * qty) / 1000.0
            mask = {int(h) for h in (appliance.get("activeHourMask") or []) if 0 <= int(h) <= 23}
            if mask:
                prepared.append(("mask", power_kw, mask))
            elif (True if appliance.get("alwaysOn") is None else bool(appliance.get("alwaysOn"))):
                prepared.append(("flat", power_kw, None))
            # else: equipo sin horario y no continuo → no aporta al pronóstico

        running_total = 0.0
        for i in range(hours):
            dt = begin + _td(hours=i)
            total_kw = 0.0
            for kind, value, mask in prepared:
                if kind == "profile":
                    total_kw += forecast_kw(value, dt)
                elif kind == "mask":
                    if dt.hour in mask:
                        total_kw += value
                else:  # flat
                    total_kw += value
            running_total += total_kw
            points.append(
                HourlyForecastPoint(datetime=dt.isoformat(), consumptionKw=round(total_kw, 4))
            )

        return ApplianceForecastSummary(
            totalConsumptionKw=round(running_total, 4),
            appliancesWithProfile=with_profile_count,
            appliancesAlwaysOn=always_on_count,
            points=points,
        )

    @strawberry.field
    def inverters(self) -> List[InverterType]:
        return [_map_inverter(inverter) for inverter in list_inverters()]

    @strawberry.field
    def inverter(self, id: str) -> Optional[InverterType]:
        inverter = get_inverter(id)
        return _map_inverter(inverter) if inverter else None

    @strawberry.field
    async def ml_predict(
        self,
        datetimes: List[str],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> List[MLPredictionType]:
        """
        Predict solar production using ML model for specific datetimes.

        Args:
            datetimes: List of ISO datetime strings (e.g., ["2025-01-15T13:00:00", "2025-01-15T14:00:00"])
            lat: Latitude (optional, defaults to system location)
            lon: Longitude (optional, defaults to system location)

        Returns:
            List of predictions with production_kw and weather features
        """
        # Get system config for location if not provided
        capacity_kw = None
        if lat is None or lon is None:
            config = get_system_config()
            lat = lat or config["location"]["lat"]
            lon = lon or config["location"]["lon"]
            capacity_kw = _get_real_capacity_kw_from_config(config)

        predictions = await predict_solar_production(datetimes, lat, lon)
        predictions = _scale_ml_predictions(predictions, capacity_kw)

        return [_to_ml_prediction_type(pred) for pred in predictions]

    @strawberry.field
    async def ml_predict_next_hours(
        self,
        hours: int = 24,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> List[MLPredictionType]:
        """
        Predict solar production for the next N hours.

        Args:
            hours: Number of hours to predict (default: 24)
            lat: Latitude (optional, defaults to system location)
            lon: Longitude (optional, defaults to system location)

        Returns:
            List of hourly predictions
        """
        # Get system config for location if not provided
        capacity_kw = None
        if lat is None or lon is None:
            config = get_system_config()
            lat = lat or config["location"]["lat"]
            lon = lon or config["location"]["lon"]
            capacity_kw = _get_real_capacity_kw_from_config(config)

        predictions = await predict_next_hours(hours, lat, lon)
        predictions = _scale_ml_predictions(predictions, capacity_kw)

        return [_to_ml_prediction_type(pred) for pred in predictions]

    @strawberry.field
    async def ml_predict_date_range(
        self,
        start_date: str,
        end_date: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> List[MLPredictionType]:
        """
        Predict solar production for all hours in a date range.

        Args:
            start_date: Start date (ISO format: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS')
            end_date: End date (ISO format: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS')
            lat: Latitude (optional, defaults to system location)
            lon: Longitude (optional, defaults to system location)

        Returns:
            List of hourly predictions for the entire date range
        """
        # Get system config for location if not provided
        capacity_kw = None
        if lat is None or lon is None:
            config = get_system_config()
            lat = lat or config["location"]["lat"]
            lon = lon or config["location"]["lon"]
            capacity_kw = _get_real_capacity_kw_from_config(config)

        predictions = await predict_for_date_range(start_date, end_date, lat, lon)
        predictions = _scale_ml_predictions(predictions, capacity_kw)

        return [_to_ml_prediction_type(pred) for pred in predictions]

    @strawberry.field
    async def ml_predict_for_hours(
        self,
        date: str,
        hours: List[int],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> List[MLPredictionType]:
        """
        Predict solar production for specific hours of a given day.

        Args:
            date: Date in YYYY-MM-DD format
            hours: List of hours (0-23) to predict for (e.g., [7, 8, 9, ..., 22] for 7am-10pm)
            lat: Optional latitude (defaults to system location)
            lon: Optional longitude (defaults to system location)

        Returns:
            List of predictions with production_kw and weather features
        """
        from .services.ml_prediction_service import predict_for_specific_hours

        config = get_system_config()
        latitude = lat if lat is not None else config["location"]["lat"]
        longitude = lon if lon is not None else config["location"]["lon"]
        capacity_kw = _get_real_capacity_kw_from_config(config)

        predictions = await predict_for_specific_hours(date, hours, latitude, longitude)
        predictions = _scale_ml_predictions(predictions, capacity_kw)

        return [
            MLPredictionType(
                datetime=pred["datetime"],
                production_kw=pred["production_kw"],
                weather=MLWeatherFeaturesType(
                    temperature_2m=pred["weather"]["temperature_2m"],
                    relative_humidity_2m=pred["weather"]["relative_humidity_2m"],
                    wind_speed_10m=pred["weather"]["wind_speed_10m"],
                    cloud_cover=pred["weather"]["cloud_cover"],
                    shortwave_radiation=pred["weather"]["shortwave_radiation"],
                ),
            )
            for pred in predictions
        ]

    @strawberry.field
    def ml_model_info(self) -> MLModelInfoType:
        """
        Get information about the loaded ML model.

        Returns:
            Model metadata including accuracy metrics and status
        """
        info = ml_model_service.get_model_info()
        return MLModelInfoType(
            loaded=info.get("loaded", False),
            model_name=info.get("model_name"),
            test_rmse=info.get("test_rmse"),
            test_r2=info.get("test_r2"),
            test_mae=info.get("test_mae"),
            features=info.get("features", []),
            training_date=info.get("training_date"),
            requires_scaling=info.get("requires_scaling"),
            reference_capacity_kw=info.get("reference_capacity_kw"),
            message=info.get("message"),
        )

    @strawberry.field
    async def battery_discharge_estimate(
        self,
        start_hour: int,
        date: Optional[str] = None,
    ) -> BatteryDischargeEstimateType:
        """
        Calculate time until battery reaches empty (0%) level.

        Simulates battery discharge/charge based on predicted production and consumption,
        starting from a given hour with batteries at 100% charge.

        Args:
            start_hour: Starting hour (0-23) to begin simulation
            date: Optional date in ISO format ('YYYY-MM-DD'). If not provided, uses today.

        Returns:
            Estimate with minutes until battery is fully discharged

        Example:
            query {
              batteryDischargeEstimate(startHour: 14) {
                minutesToEmpty
                startHour
                batteryCapacityKwh
              }
            }
        """
        result = await calculate_battery_discharge_time(start_hour, date)
        return BatteryDischargeEstimateType(
            minutesToEmpty=result["minutesToEmpty"],
            startHour=result["startHour"],
            batteryCapacityKwh=result["batteryCapacityKwh"],
        )

    @strawberry.field
    def invitation_codes(self, info: strawberry.types.Info) -> List[InvitationCodeType]:
        require_admin(info.context)
        codes = list_invitation_codes()
        return [
            InvitationCodeType(
                id_=code["_id"],
                code=code["code"],
                role=code["role"],
                isUsed=code["isUsed"],
                createdBy=code.get("createdBy"),
                usedBy=code.get("usedBy"),
                createdAt=code.get("createdAt"),
                updatedAt=code.get("updatedAt"),
            )
            for code in codes
        ]

    @strawberry.field
    def users(self, info: strawberry.types.Info) -> List[UserType]:
        require_admin(info.context)
        return [_map_user(user) for user in list_users()]

    @strawberry.field
    def active_sessions(self, info: strawberry.types.Info) -> List[SessionType]:
        require_admin(info.context)
        sessions = list_active_sessions()
        return [
            SessionType(
                id_=s["_id"],
                email=s["email"],
                ip=s["ip"],
                userAgent=s["userAgent"],
                deviceType=s["deviceType"],
                createdAt=s.get("createdAt"),
                expiresAt=s.get("expiresAt"),
            )
            for s in sessions
        ]

    @strawberry.field
    def ldap_config(self, info: strawberry.types.Info) -> LdapConfigType:
        """Current LDAP configuration (admin only). Bind password is never returned."""
        require_admin(info.context)
        return _map_ldap_config(get_ldap_config())

    @strawberry.field
    def ldap_enabled(self) -> bool:
        """Public flag so the login screen knows whether to show the LDAP tab."""
        return is_ldap_enabled()

    @strawberry.field
    def historical_readings(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 288,
    ) -> List[HistoricalReadingType]:
        """Query historical solar production readings with optional date range filter."""
        readings = get_readings(start_date, end_date, limit)
        return [
            HistoricalReadingType(
                id_=r["_id"],
                timestamp=r["timestamp"],
                productionKw=r.get("productionKw", r.get("production", 0.0)),
            )
            for r in readings
        ]

    @strawberry.field
    def latest_panel_cleanliness(self) -> Optional[PanelCleanlinessType]:
        """Última lectura de limpieza de paneles, o None si aún no hay ninguna."""
        reading = get_latest_cleanliness()
        if not reading:
            return None
        return PanelCleanlinessType(
            id_=reading["_id"],
            timestamp=reading["timestamp"],
            clasificacion=reading["clasificacion"],
            porcentajeLimpio=reading["porcentajeLimpio"],
            porcentajeSucio=reading["porcentajeSucio"],
            source=reading["source"],
        )

    @strawberry.field
    def daily_summaries(self, days: int = 30) -> List[DailySummaryType]:
        """Get daily aggregated production summaries for the last N days."""
        summaries = get_daily_summaries(days)
        return [DailySummaryType(**s) for s in summaries]

    @strawberry.field
    def appliance_batches(self, appliance_id: str) -> List[ApplianceBatchType]:
        """List all uploaded measurement batches for an appliance."""
        batches = list_appliance_batches(appliance_id)
        return [ApplianceBatchType(**b) for b in batches]

    @strawberry.field
    def daily_report(self, date: str) -> DailyReportType:
        """
        Full daily energy report for a given date (YYYY-MM-DD).
        Combines real Hioki measurements with estimated consumption from appliance configs.
        """
        report = get_daily_report(date)
        appliances = [
            DailyReportApplianceType(
                applianceId=a["applianceId"],
                name=a["name"],
                mode=a["mode"],
                kwhDay=a["kwhDay"],
                kwhDayEstimated=a.get("kwhDayEstimated"),
                errorPercent=a.get("errorPercent"),
                readingCount=a.get("readingCount", 0),
            )
            for a in report["appliances"]
        ]
        return DailyReportType(
            date=report["date"],
            productionKwh=report.get("productionKwh"),
            measuredConsumptionKwh=report["measuredConsumptionKwh"],
            estimatedConsumptionKwh=report["estimatedConsumptionKwh"],
            totalConsumptionKwh=report["totalConsumptionKwh"],
            hasRealData=report["hasRealData"],
            appliances=appliances,
        )

    @strawberry.field
    async def monthly_production(self, year: Optional[int] = None) -> List[MonthlyProductionType]:
        """
        Producción por mes del año, reconstruida con el modelo de producción.

        Mezcla, en un único pastel anual:
          * meses ya pasados → clima REAL de Open-Meteo (Archive API / forecast);
          * meses futuros    → climatología (mismo mes del año anterior);
          * mes en curso      → parte real + parte estimada (``mixto``).

        Reutiliza el mismo modelo ML que el resto de la pantalla; aquí solo se
        orquestan los tramos y se totaliza por mes.
        """
        import asyncio
        from datetime import date as _date, datetime as _dt, timedelta as _td, timezone as _tz

        MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

        config = get_system_config()
        lat = config["location"]["lat"]
        lon = config["location"]["lon"]
        capacity_kw = _get_real_capacity_kw_from_config(config)

        today = _date.today()
        target_year = year or today.year

        kwh = {m: 0.0 for m in range(1, 13)}
        has_real = {m: False for m in range(1, 13)}
        has_pred = {m: False for m in range(1, 13)}

        async def _fetch_chunk(start: "_date", end: "_date", is_pred: bool):
            """Predice un tramo y devuelve (preds, is_pred). NO muta el acumulador:
            la fusión se hace tras el gather, así no hay carrera entre tramos."""
            if start > end:
                return ([], is_pred)
            # Horas UTC-aware (no naïve): así el pipeline no intenta localizar a
            # hora local y evitamos el hueco del cambio de horario de verano, que
            # rompería pandas al cruzar marzo en un rango anual.
            cur = _dt(start.year, start.month, start.day, tzinfo=_tz.utc)
            stop = _dt(end.year, end.month, end.day, 23, tzinfo=_tz.utc)
            datetimes = []
            while cur <= stop:
                datetimes.append(cur.isoformat())
                cur += _td(hours=1)
            preds = await predict_solar_production(datetimes, lat, lon)
            preds = _scale_ml_predictions(preds, capacity_kw)
            return (preds, is_pred)

        # Cada tramo es una petición HTTP independiente a Open-Meteo (rama "fuera de
        # ventana" del caché, sin estado compartido), así que se lanzan en paralelo
        # con seguridad: el cuello de botella es la red, y el pastel baja de ~10 s a
        # ~4 s. La inferencia del modelo sigue serializada (es síncrona, no awaita).
        tasks = []
        if target_year < today.year:
            # Año completamente pasado → todo histórico real.
            tasks.append(_fetch_chunk(_date(target_year, 1, 1), _date(target_year, 12, 31), False))
        elif target_year > today.year:
            # Año completamente futuro → climatología del último año conocido.
            tasks.append(_fetch_chunk(_date(today.year - 1, 1, 1), _date(today.year - 1, 12, 31), True))
        else:
            # Año en curso. El tramo real se parte en dos para respetar el ruteo de
            # Open-Meteo (Archive para lo viejo, /forecast para los últimos días).
            archive_end = today - _td(days=6)
            tasks.append(_fetch_chunk(_date(today.year, 1, 1), archive_end, False))          # real (archivo)
            tasks.append(_fetch_chunk(max(_date(today.year, 1, 1), today - _td(days=5)), today, False))  # real (forecast)
            # Lo que falta del año → mismas fechas del año anterior (climatología).
            rem_start = today + _td(days=1)
            if rem_start <= _date(today.year, 12, 31):
                tasks.append(_fetch_chunk(rem_start.replace(year=today.year - 1),
                                          _date(today.year - 1, 12, 31), True))

        chunks = await asyncio.gather(*tasks)
        # Fusión secuencial (sin concurrencia) → sin condiciones de carrera.
        for preds, is_pred in chunks:
            for p in preds:
                m = _dt.fromisoformat(p["datetime"]).month
                kwh[m] += p["production_kw"]
                if is_pred:
                    has_pred[m] = True
                else:
                    has_real[m] = True

        result: List[MonthlyProductionType] = []
        for m in range(1, 13):
            if has_real[m] and has_pred[m]:
                source = "mixto"
            elif has_pred[m]:
                source = "prediccion"
            else:
                source = "historico"
            result.append(MonthlyProductionType(
                month=m,
                monthName=MESES[m - 1],
                productionKwh=round(kwh[m], 1),
                source=source,
            ))
        return result

    @strawberry.field
    def appliance_readings(
        self, appliance_id: str, start_date: str, end_date: str
    ) -> List[ApplianceReadingPointType]:
        """Return minute-level Hioki readings for an appliance in a date range."""
        points = get_appliance_readings(appliance_id, start_date, end_date)
        return [ApplianceReadingPointType(**p) for p in points]

    @strawberry.field
    def weather_sources(self) -> List[WeatherSourceType]:
        return [_map_weather_source(item) for item in list_weather_sources()]

    @strawberry.field
    def active_weather_source(self) -> Optional[WeatherSourceType]:
        source = get_active_weather_source()
        return _map_weather_source(source) if source else None

    @strawberry.field
    def location_config(self) -> LocationConfigExtType:
        """Return the current location configuration."""
        data = get_location_config()
        return LocationConfigExtType(
            lat=data["lat"],
            lon=data["lon"],
            name=data["name"],
            updatedAt=data.get("updatedAt"),
        )

    @strawberry.field
    def shadow_profile(self) -> Optional[ShadowProfileType]:
        """Return the saved hourly shadow profile, or null if none has been stored."""
        data = get_shadow_profile()
        if data is None:
            return None
        return ShadowProfileType(
            slots=[
                ShadowSlotType(
                    hour=s["hour"],
                    shadow_pct=s["shadowPct"],
                    prod_override=s.get("prodOverride"),
                )
                for s in data["slots"]
            ],
            avg_shadow=data["avgShadow"],
            avg_prod=data["avgProd"],
            updated_at=data.get("updatedAt"),
        )


# ============================================================================
# Inputs
# ============================================================================


@strawberry.input
class PanelInput:
    manufacturer: str
    model: Optional[str] = None
    ratedPowerKw: float
    quantity: int
    tiltDegrees: Optional[float] = None
    orientation: Optional[str] = None
    efficiencyPercent: Optional[float] = None
    areaM2: Optional[float] = None


@strawberry.input
class BatteryInput:
    manufacturer: str
    model: Optional[str] = None
    capacityKwh: float
    quantity: int
    maxDepthOfDischargePercent: Optional[float] = None
    chargeRateKw: Optional[float] = None
    dischargeRateKw: Optional[float] = None
    efficiencyPercent: Optional[float] = None


@strawberry.input
class InverterInput:
    manufacturer: str
    model: Optional[str] = None
    ratedPowerKw: float
    quantity: int
    efficiencyPercent: Optional[float] = None


@strawberry.input
class ApplianceModeInput:
    name: str
    averagePowerW: float
    maxPowerW: Optional[float] = None


@strawberry.input
class ApplianceInput:
    name: str
    category: Optional[str] = None
    averagePowerW: Optional[float] = None
    maxPowerW: Optional[float] = None
    measuredPowerW: Optional[float] = None
    quantity: int
    activeHours: Optional[float] = None
    selectedModeIndex: Optional[int] = None
    modes: Optional[List[ApplianceModeInput]] = None
    alwaysOn: Optional[bool] = True
    useMeasurements: Optional[bool] = False
    activeHourMask: Optional[List[int]] = None
    uncoveredHoursFill: Optional[str] = None


@strawberry.input
class RegisterInput:
    email: str
    password: str
    invitationCode: str
    name: Optional[str] = None
    # role removed, determined by code


@strawberry.input
class LoginInput:
    email: str
    password: str


@strawberry.input
class ChangePasswordInput:
    currentPassword: str
    newPassword: str


@strawberry.input
class LdapLoginInput:
    email: str
    password: str
    invitationCode: Optional[str] = None


@strawberry.input
class WeatherSourceInput:
    name: str
    baseUrl: Optional[str] = None
    authType: Optional[str] = "none"
    authHeaderName: Optional[str] = None
    authQueryName: Optional[str] = None
    authValue: Optional[str] = None
    queryParams: Optional[JSON] = None
    fieldMapping: Optional[JSON] = None
    locationName: Optional[str] = None
    enabled: Optional[bool] = True
    isActive: Optional[bool] = False


@strawberry.input
class LocationConfigInput:
    lat: float
    lon: float
    name: str


@strawberry.input
class LdapConfigInput:
    enabled: bool = False
    serverUrl: Optional[str] = None
    baseDn: Optional[str] = None
    bindDn: Optional[str] = None
    bindPassword: Optional[str] = None
    userSearchFilter: Optional[str] = None
    emailAttr: Optional[str] = None
    nameAttr: Optional[str] = None
    useTls: Optional[bool] = False
    connectTimeout: Optional[int] = None
    # Solo se usan en testLdapConnection (probar credenciales de muestra).
    sampleEmail: Optional[str] = None
    samplePassword: Optional[str] = None


# ============================================================================
# Mutations
# ============================================================================


@strawberry.type
class Mutation:
    @strawberry.mutation(name="createPanel")
    def create_panel_mutation(self, info: strawberry.types.Info, input: PanelInput) -> PanelType:
        require_admin(info.context)
        panel = create_panel(input.__dict__)
        return _map_panel(panel)

    @strawberry.mutation(name="updatePanel")
    def update_panel_mutation(self, info: strawberry.types.Info, id: str, input: PanelInput) -> PanelType:
        require_admin(info.context)
        panel = update_panel(id, input.__dict__)
        if not panel:
            raise ValueError("Panel no encontrado.")
        return _map_panel(panel)

    @strawberry.mutation(name="deletePanel")
    def delete_panel_mutation(self, info: strawberry.types.Info, id: str) -> bool:
        require_admin(info.context)
        return delete_panel(id)

    @strawberry.mutation(name="createBattery")
    def create_battery_mutation(self, info: strawberry.types.Info, input: BatteryInput) -> BatteryType:
        require_admin(info.context)
        battery = create_battery(input.__dict__)
        return _map_battery(battery)

    @strawberry.mutation(name="updateBattery")
    def update_battery_mutation(self, info: strawberry.types.Info, id: str, input: BatteryInput) -> BatteryType:
        require_admin(info.context)
        battery = update_battery(id, input.__dict__)
        if not battery:
            raise ValueError("Batería no encontrada.")
        return _map_battery(battery)

    @strawberry.mutation(name="deleteBattery")
    def delete_battery_mutation(self, info: strawberry.types.Info, id: str) -> bool:
        require_admin(info.context)
        return delete_battery(id)

    @strawberry.mutation(name="createAppliance")
    def create_appliance_mutation(self, info: strawberry.types.Info, input: ApplianceInput) -> ApplianceType:
        require_admin(info.context)
        payload = {
            **input.__dict__,
            "modes": [mode.__dict__ for mode in input.modes] if input.modes else [],
        }
        appliance = create_appliance(payload)
        return _map_appliance(appliance)

    @strawberry.mutation(name="updateAppliance")
    def update_appliance_mutation(self, info: strawberry.types.Info, id: str, input: ApplianceInput) -> ApplianceType:
        require_admin(info.context)
        payload = {
            **input.__dict__,
            "modes": [mode.__dict__ for mode in input.modes] if input.modes else [],
        }
        appliance = update_appliance(id, payload)
        if not appliance:
            raise ValueError("Electrodoméstico no encontrado.")
        return _map_appliance(appliance)

    @strawberry.mutation(name="deleteAppliance")
    def delete_appliance_mutation(self, info: strawberry.types.Info, id: str) -> bool:
        require_admin(info.context)
        return delete_appliance(id)

    @strawberry.mutation(name="uploadApplianceMeasurement")
    def upload_appliance_measurement_mutation(
        self, info: strawberry.types.Info, id: str, fileContent: str
    ) -> ApplianceType:
        """
        Attach a power-meter export (TSV/CSV) to an appliance. The file is
        parsed and converted into a 168-hour (weekday x hour) average kW
        profile used to forecast future consumption.
        """
        require_admin(info.context)
        appliance = attach_measurement(id, fileContent)
        return _map_appliance(appliance)

    @strawberry.mutation(name="clearApplianceMeasurement")
    def clear_appliance_measurement_mutation(self, info: strawberry.types.Info, id: str) -> ApplianceType:
        require_admin(info.context)
        appliance = clear_measurement(id)
        if not appliance:
            raise ValueError("Electrodoméstico no encontrado.")
        return _map_appliance(appliance)

    @strawberry.mutation(name="previewApplianceBatch")
    def preview_appliance_batch_mutation(
        self, info: strawberry.types.Info, file_content: str
    ) -> BatchPreviewType:
        """Parse a file and return detected date range + sample count without storing."""
        require_admin(info.context)
        result = preview_batch(file_content)
        return BatchPreviewType(**result)

    @strawberry.mutation(name="uploadApplianceBatch")
    def upload_appliance_batch_mutation(
        self,
        info: strawberry.types.Info,
        id: str,
        file_content: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> ApplianceBatchType:
        """
        Upload a Hioki measurement file for an appliance. Accumulates readings
        into mediciones_equipos (upsert) and rebuilds the 168-bin hourly profile.
        Stores a snapshot of estimated kWh/day for error analysis.
        """
        require_admin(info.context)
        # Compute estimation snapshot from current appliance configs
        this_appl = get_appliance(id)
        if not this_appl:
            raise ValueError("Electrodoméstico no encontrado.")

        all_appliances = list_appliances()
        kwh_this = (
            (this_appl.get("averagePowerW") or 0.0)
            * (this_appl.get("activeHours") or 0.0)
            / 1000.0
        )
        kwh_others = sum(
            (a.get("averagePowerW") or 0.0) * (a.get("activeHours") or 0.0) / 1000.0
            for a in all_appliances
            if str(a.get("_id", "")) != id
        )

        batch = upload_appliance_batch(
            appliance_id=id,
            appliance_name=this_appl.get("name", ""),
            file_content=file_content,
            filename="archivo.xls",
            start_date=start_date,
            end_date=end_date,
            kwh_day_estimated_this=kwh_this,
            kwh_day_estimated_others=kwh_others,
        )
        return ApplianceBatchType(**batch)

    @strawberry.mutation(name="deleteApplianceBatch")
    def delete_appliance_batch_mutation(
        self, info: strawberry.types.Info, batch_id: str
    ) -> bool:
        """Delete a measurement batch and rebuild the appliance's hourly profile."""
        require_admin(info.context)
        return delete_appliance_batch(batch_id)

    @strawberry.mutation(name="createInverter")
    def create_inverter_mutation(self, info: strawberry.types.Info, input: InverterInput) -> InverterType:
        require_admin(info.context)
        inverter = create_inverter(input.__dict__)
        return _map_inverter(inverter)

    @strawberry.mutation(name="updateInverter")
    def update_inverter_mutation(self, info: strawberry.types.Info, id: str, input: InverterInput) -> InverterType:
        require_admin(info.context)
        inverter = update_inverter(id, input.__dict__)
        if not inverter:
            raise ValueError("Inversor no encontrado.")
        return _map_inverter(inverter)

    @strawberry.mutation(name="deleteInverter")
    def delete_inverter_mutation(self, info: strawberry.types.Info, id: str) -> bool:
        require_admin(info.context)
        return delete_inverter(id)

    @strawberry.mutation(name="registerUser")
    def register_user_mutation(self, info: strawberry.types.Info, input: RegisterInput) -> AuthPayloadType:
        import uuid
        from datetime import timedelta
        from app.config import settings as _s
        user = register_user(input.__dict__)
        jti = uuid.uuid4().hex
        token = create_token(user["email"], user["role"], jti)
        from datetime import datetime as _dt
        req = info.context.get("request")
        ip = (req.headers.get("x-forwarded-for") or (req.client.host if req.client else "")) if req else ""
        ua = req.headers.get("user-agent", "") if req else ""
        create_session(user["email"], ip, ua, jti, _dt.utcnow() + timedelta(days=_s.JWT_EXPIRE_DAYS))
        return AuthPayloadType(user=_map_user(user), token=token)

    @strawberry.mutation(name="loginUser")
    def login_user_mutation(self, info: strawberry.types.Info, input: LoginInput) -> AuthPayloadType:
        import uuid
        from datetime import timedelta, datetime as _dt
        from app.config import settings as _s
        user = authenticate_user(input.__dict__)
        jti = uuid.uuid4().hex
        token = create_token(user["email"], user["role"], jti)
        req = info.context.get("request")
        ip = (req.headers.get("x-forwarded-for") or (req.client.host if req.client else "")) if req else ""
        ua = req.headers.get("user-agent", "") if req else ""
        create_session(user["email"], ip, ua, jti, _dt.utcnow() + timedelta(days=_s.JWT_EXPIRE_DAYS))
        return AuthPayloadType(user=_map_user(user), token=token)

    @strawberry.mutation(name="changePassword")
    def change_password_mutation(self, info: strawberry.types.Info, input: ChangePasswordInput) -> bool:
        # The email comes from the verified JWT, never from the client payload.
        current_user = require_auth(info.context)
        return change_password(current_user["sub"], input.currentPassword, input.newPassword)

    @strawberry.mutation(name="loginLdap")
    def login_ldap_mutation(self, info: strawberry.types.Info, input: LdapLoginInput) -> AuthPayloadType:
        import uuid
        from datetime import timedelta, datetime as _dt
        from app.config import settings as _s
        user = authenticate_or_provision_ldap(input.__dict__)
        jti = uuid.uuid4().hex
        token = create_token(user["email"], user["role"], jti)
        req = info.context.get("request")
        ip = (req.headers.get("x-forwarded-for") or (req.client.host if req.client else "")) if req else ""
        ua = req.headers.get("user-agent", "") if req else ""
        create_session(user["email"], ip, ua, jti, _dt.utcnow() + timedelta(days=_s.JWT_EXPIRE_DAYS))
        return AuthPayloadType(user=_map_user(user), token=token)

    @strawberry.mutation(name="revokeSession")
    def revoke_session_mutation(self, info: strawberry.types.Info, id: str) -> bool:
        require_admin(info.context)
        return revoke_session(id)

    @strawberry.mutation(name="deleteUser")
    def delete_user_mutation(self, info: strawberry.types.Info, id: str) -> bool:
        current_admin = require_admin(info.context)
        target_users = [u for u in list_users() if u["_id"] == id]
        if not target_users:
            raise Exception("Usuario no encontrado.")
        if target_users[0]["email"] == current_admin["sub"]:
            raise Exception("No puedes eliminar tu propia cuenta.")
        revoke_sessions_by_email(target_users[0]["email"])
        return delete_user(id)

    @strawberry.mutation(name="generateInvitationCode")
    def generate_invitation_code_mutation(self, info: strawberry.types.Info, role: str, createdBy: str) -> InvitationCodeType:
        require_admin(info.context)
        code = create_invitation_code(role, createdBy)
        return InvitationCodeType(
            id_=code["_id"],
            code=code["code"],
            role=code["role"],
            isUsed=code["isUsed"],
            createdBy=code.get("createdBy"),
            usedBy=code.get("usedBy"),
            createdAt=code.get("createdAt"),
            updatedAt=code.get("updatedAt"),
        )

    @strawberry.mutation(name="saveWeatherSource")
    def save_weather_source_mutation(
        self,
        info: strawberry.types.Info,
        input: WeatherSourceInput,
        id: Optional[str] = None,
    ) -> WeatherSourceType:
        require_admin(info.context)
        payload = input.__dict__
        source = save_weather_source(payload, id)
        return _map_weather_source(source)

    @strawberry.mutation(name="deleteWeatherSource")
    def delete_weather_source_mutation(self, info: strawberry.types.Info, id: str) -> bool:
        require_admin(info.context)
        return delete_weather_source(id)

    @strawberry.mutation(name="setActiveWeatherSource")
    def set_active_weather_source_mutation(self, info: strawberry.types.Info, id: str) -> bool:
        require_admin(info.context)
        return set_active_weather_source(id)

    @strawberry.mutation(name="testWeatherSource")
    async def test_weather_source_mutation(
        self,
        info: strawberry.types.Info,
        input: WeatherSourceInput,
        useMock: bool = False,
    ) -> WeatherSourceTestResultType:
        require_admin(info.context)
        config = get_system_config()
        result = await test_weather_source(
            source_payload=input.__dict__,
            lat=config["location"]["lat"],
            lon=config["location"]["lon"],
            location_name=config["location"]["name"],
            use_mock=useMock,
        )
        return WeatherSourceTestResultType(
            success=result.get("success", False),
            message=result.get("message") or "Sin respuesta",
            fields=[
                WeatherFieldCandidateType(
                    path=item.get("path", ""),
                    valueType=item.get("valueType", ""),
                    sampleValue=item.get("sampleValue", ""),
                )
                for item in result.get("fields", [])
            ],
            rawJson=result.get("rawJson") or "{}",
        )

    @strawberry.mutation(name="saveLdapConfig")
    def save_ldap_config_mutation(
        self, info: strawberry.types.Info, input: LdapConfigInput
    ) -> LdapConfigType:
        """Persist the LDAP connection settings. Admin only."""
        require_admin(info.context)
        cfg = save_ldap_config(input.__dict__)
        return _map_ldap_config(cfg)

    @strawberry.mutation(name="testLdapConnection")
    async def test_ldap_connection_mutation(
        self, info: strawberry.types.Info, input: LdapConfigInput
    ) -> LdapTestResultType:
        """
        Probe the directory with the supplied (or stored) settings, optionally
        verifying a sample login. Admin only. Never throws on connection failure.
        """
        require_admin(info.context)
        result = await test_ldap_connection(input.__dict__)
        return LdapTestResultType(
            success=bool(result.get("success")),
            message=result.get("message") or "",
            sampleUser=result.get("sampleUser"),
        )

    @strawberry.mutation(name="seedHistoricalData")
    def seed_historical_data_mutation(
        self,
        info: strawberry.types.Info,
        days: int = 30,
    ) -> int:
        """Seed historical readings for demo/thesis purposes. Admin only."""
        require_admin(info.context)
        return seed_historical_data(days)

    @strawberry.mutation(name="saveLocationConfig")
    def save_location_config_mutation(
        self,
        info: strawberry.types.Info,
        input: LocationConfigInput,
    ) -> LocationConfigExtType:
        """Update system location. Admin only."""
        require_admin(info.context)
        data = save_location_config(input.lat, input.lon, input.name)
        return LocationConfigExtType(
            lat=data["lat"],
            lon=data["lon"],
            name=data["name"],
            updatedAt=data.get("updatedAt"),
        )

    @strawberry.mutation(name="resetSystemData")
    def reset_system_data_mutation(self, info: strawberry.types.Info) -> bool:
        """
        Borra toda la configuración del sistema (paneles, baterías, inversores,
        electrodomésticos, ubicación y perfil de consumo) para permitir
        volver a ejecutar el asistente de configuración. Solo administradores.
        """
        require_admin(info.context)
        db = get_database()
        db["paneles"].delete_many({})
        db["baterias"].delete_many({})
        db["inversores"].delete_many({})
        db["electrodomesticos"].delete_many({})
        db["ubicacion_config"].delete_many({})
        db["shadow_profile"].delete_many({})
        return True

    @strawberry.mutation(name="saveShadowProfile")
    def save_shadow_profile_mutation(
        self,
        info: strawberry.types.Info,
        slots: List[ShadowSlotInput],
    ) -> ShadowProfileType:
        """
        Persist the hourly shadow profile for the installation.
        Only day-lit slots (elevation > 0) should be included.
        Admin only.
        """
        require_admin(info.context)
        raw = [
            {"hour": s.hour, "shadowPct": s.shadow_pct, "prodOverride": s.prod_override}
            for s in slots
        ]
        data = save_shadow_profile(raw)
        return ShadowProfileType(
            slots=[
                ShadowSlotType(
                    hour=sl["hour"],
                    shadow_pct=sl["shadowPct"],
                    prod_override=sl.get("prodOverride"),
                )
                for sl in data["slots"]
            ],
            avg_shadow=data["avgShadow"],
            avg_prod=data["avgProd"],
            updated_at=data.get("updatedAt"),
        )


# ============================================================================
# Schema
# ============================================================================

schema = strawberry.Schema(query=Query, mutation=Mutation)
