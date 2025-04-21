
# Housing Price Prediction API

A simple Flask-based API that predicts housing prices using a trained regression model. This project is containerized using Docker for easy deployment.

## Features

- Accepts multiple input records (each with 12 features)
- Returns predicted housing prices
- Includes health check endpoint

## Project Structure

```
housing_app/
│
├── app.py                  # Flask API logic
├── housing_model.pkl       # Trained regression model
├── scaler.pkl              # Scaler used during training
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build configuration
└── README.md               # You're here!
```

## Setup & Run

### 1. Build Docker Image

From inside the `housing_app/` directory, run the following command to build the Docker image:

```bash
docker build -t housing-api .
```

### 2. Run Docker Container

Run the Docker container:

```bash
docker run -d -p 9000:9000 --name housing-app housing-api
```

### 3. Check Health

To check if the API is live, run:

```bash
curl http://localhost:9000/health
```

Expected output:

```json
{"status": "ok"}
```

## Usage

### POST `/predict`

#### Input Example:

```json
{
  "features": [
    [7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1, 1],
    [8960, 4, 4, 4, 1, 0, 0, 0, 1, 3, 0, 1]
  ]
}
```

Each input must be a list of **12 numeric values** in the following order:

- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `parking`
- `prefarea`
- `furnishingstatus`

#### Output Example:

```json
{
  "predictions": [1234567.0, 8901234.5],
  "confidences": [0.95, 0.89]
}
```

### Notes

- All categorical fields are encoded as integers:
  - `yes` → 1, `no` → 0
  - `furnishingstatus`: `furnished` → 1, `semi-furnished` → 2, `unfurnished` → 3
- Input validation is handled. Malformed inputs return HTTP 400 with an error message.

## Stopping & Cleaning Up

To stop and remove the Docker container:

```bash
docker stop housing-app
docker rm housing-app
```

---

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

