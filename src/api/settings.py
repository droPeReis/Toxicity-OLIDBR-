from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    API_NAME: str = Field("ToChiquinho", description="API name")
    API_VERSION: str = Field("0.1.0", description="API version")
    API_DESCRIPTION: str = Field(
        "Toxicity Detection system", description="API description"
    )
    API_PORT: int = Field(80, description="API port")
    API_HEALTHCHECK_PATH: str = Field(
        "/health", description="API healthcheck path"
    )
    API_TASK: str = Field("all", description="API task that will be served")

    API_ROUTE_TOXICITY_ENDPOINT: str = Field(
        None,
        description="API endpoint to toxicity classification. (e.g. http://localhost:8002/predict)",
    )
    API_ROUTE_TOXICITY_TARGET_ENDPOINT: str = Field(
        None,
        description="API endpoint to toxicity target classification. (e.g. http://localhost:8004/predict)",
    )
    API_ROUTE_TOXICITY_TARGET_TYPE_ENDPOINT: str = Field(
        None,
        description="API endpoint to toxicity target type identification. (e.g. http://localhost:8005/predict)",
    )
    API_ROUTE_TOXICITY_TYPE_ENDPOINT: str = Field(
        None,
        description="API endpoint to toxicity type detection. (e.g. http://localhost:8003/predict)",
    )
    API_ROUTE_TOXIC_SPANS_ENDPOINT: str = Field(
        None,
        description="API endpoint to toxic spans detection (e.g. http://localhost:8001/predict)",
    )

    @validator("API_TASK")
    def validate_api_task(cls, v):
        tasks = [
            "all",
            "route",
            "toxic_spans",
            "toxicity",
            "toxicity_target",
            "toxicity_target_type",
            "toxicity_type",
        ]

        if v not in tasks:
            raise ValueError(f"API_TASK must be one of {tasks}.")

        if v == "route":
            if not cls.API_ROUTE_TOXIC_SPANS_ENDPOINT:
                raise ValueError("API_ROUTE_TOXIC_SPANS_ENDPOINT must be set.")
            if not cls.API_ROUTE_TOXICITY_ENDPOINT:
                raise ValueError("API_ROUTE_TOXICITY_ENDPOINT must be set.")
            if not cls.API_ROUTE_TOXICITY_TARGET_ENDPOINT:
                raise ValueError(
                    "API_ROUTE_TOXICITY_TARGET_ENDPOINT must be set."
                )
            if not cls.API_ROUTE_TOXICITY_TARGET_TYPE_ENDPOINT:
                raise ValueError(
                    "API_ROUTE_TOXICITY_TARGET_TYPE_ENDPOINT must be set."
                )
            if not cls.API_ROUTE_TOXICITY_TYPE_ENDPOINT:
                raise ValueError(
                    "API_ROUTE_TOXICITY_TYPE_ENDPOINT must be set."
                )

        return v
