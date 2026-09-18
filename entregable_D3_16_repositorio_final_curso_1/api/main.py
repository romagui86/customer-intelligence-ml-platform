from io import StringIO
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from api.dependencies import (
    get_model,
    get_prediction_threshold,
)
from api.schemas import (
    CustomerInput,
    PredictionResponse,
)
from src.models.prediction import predict_customers


app = FastAPI(
    title="NovaTel Customer Churn API",
    description=(
        "API REST para predicción de churn "
        "de clientes de NovaTel."
    ),
    version="1.0.0",
)

PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total number of churn predictions",
    ["prediction"],
)

PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Latency of individual churn predictions",
)


@app.get("/")
def root():
    return {
        "service": "customer-churn-api",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    get_model()
    return {
        "status": "healthy",
        "service": "customer-churn-api",
    }


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerInput):
    try:
        start_time = perf_counter()

        model = get_model()
        threshold = get_prediction_threshold()

        dataframe = pd.DataFrame(
            [customer.model_dump()]
        )

        result = predict_customers(
            model=model,
            dataframe=dataframe,
            threshold=threshold,
        )

        row = result.iloc[0]

        prediction = int(
            row["churn_prediction"]
        )

        PREDICTIONS_TOTAL.labels(
            prediction=str(prediction)
        ).inc()

        PREDICTION_LATENCY.observe(
            perf_counter() - start_time
        )

        return PredictionResponse(
            customer_id=(
                str(row["customer_id"])
                if "customer_id" in result.columns
                else None
            ),
            churn_probability=float(
                row["churn_probability"]
            ),
            churn_prediction=prediction,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The CSV file is empty.",
        )

    try:
        dataframe = pd.read_csv(
            StringIO(content.decode("utf-8"))
        )

        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="The CSV file is empty.",
            )

        result = predict_customers(
            model=get_model(),
            dataframe=dataframe,
            threshold=get_prediction_threshold(),
        )

        output = StringIO()
        result.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    'attachment; filename="predictions.csv"'
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
