# Tech Talent Salary Prediction

- **Live API (Swagger UI):** [https://tech-salary-api.onrender.com/docs](https://tech-salary-api.onrender.com/docs)
- **Predict endpoint (POST):** [https://tech-salary-api.onrender.com/predict](https://tech-salary-api.onrender.com/predict)
- **Video demo:** [https://youtu.be/REPLACE_WITH_YOUR_VIDEO](https://youtu.be/REPLACE_WITH_YOUR_VIDEO)

## Mission and problem
My mission is to use technology to tackle unemployment, education and access to tech.
This project predicts a tech professional's expected annual salary (USD) from their profile,
so a talent platform can show members their market value and what upskilling is worth.
It is a regression problem with a continuous salary target.

## Dataset
The data is the Data Science / Tech Job Salaries dataset from Kaggle:
https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023
A copy is included at
`summative/linear_regression/data/ds_salaries.csv`. It has 3,755 rows and 11 columns, covering
93 job titles across 78 countries for the years 2020 to 2023, with four experience levels, four
employment types and three company sizes. That gives it both volume and variety. The target
column is `salary_in_usd`.

## Visualisations
The notebook (`summative/linear_regression/multivariate.ipynb`) contains the visualisations with
their outputs saved inline, so they render on GitHub without rerunning anything:

1. Distribution of the salary target (histogram and box plot). Salaries are skewed to the right,
   which pulls a linear model toward the high earners.
2. Salary by experience level and by company size (box plots). Pay rises with both, and both have
   a natural order, so they are ordinal encoded.
3. Correlation heatmap. Experience level has the strongest link with salary, so it is the lead
   feature. Remote ratio is close to zero, so it carries little weight.

## Feature engineering
- Dropped `salary` and `salary_currency` because `salary_in_usd` is computed from them, so keeping
  them would leak the target.
- Dropped `employee_residence` because it repeats `company_location`.
- Grouped the 93 job titles into 6 role families (`job_category`) instead of one hot encoding all 93.
- Kept the top 6 countries and bucketed the rest as Other, since the US alone is 81% of the rows.
- Ordinal encoded `experience_level` and `company_size` because they have a natural order.
- One hot encoded `employment_type` and the grouped columns, and standardised the numeric columns.

The model ends up using 7 input features, which is exactly what the API and the app expose. All the
preprocessing is inside the saved pipeline, so the API only needs the raw feature values.

## Models and results
Four models were compared using RMSE as the loss metric (lower is better): two gradient descent
linear regressors (scikit-learn `SGDRegressor` and a batch gradient descent version written from
scratch), a Random Forest, and a Decision Tree.

| Model | Type | Test RMSE (USD) | Test MAE | Test R2 |
|---|---|---|---|---|
| SGD Linear Regression | linear, stochastic GD | 48,645 | 37,236 | 0.40 |
| Batch GD Linear Regression | linear, batch GD from scratch | 48,672 | 37,269 | 0.40 |
| Random Forest | ensemble | 48,968 | 37,241 | 0.39 |
| Decision Tree | tree | 49,817 | 37,934 | 0.37 |

The scores are close. The SGD linear model has the lowest test RMSE and the smallest gap between
train and test, so it is saved as `summative/API/model/best_model.pkl` and served by the API.

## Public API
The API is deployed and publicly reachable on Render:

- Base URL: [https://tech-salary-api.onrender.com](https://tech-salary-api.onrender.com)
- Swagger UI: [https://tech-salary-api.onrender.com/docs](https://tech-salary-api.onrender.com/docs)
- Predict endpoint (POST): [https://tech-salary-api.onrender.com/predict](https://tech-salary-api.onrender.com/predict)
- Health check: [https://tech-salary-api.onrender.com/health](https://tech-salary-api.onrender.com/health)

The free tier sleeps when idle, so the first request after a while can take about 30 to 60 seconds
to wake up.

Example request body:
```json
{
  "work_year": 2023, "experience_level": "SE", "employment_type": "FT",
  "job_category": "Data Scientist", "company_size": "M",
  "company_location_grp": "US", "remote_ratio": 100
}
```

## Video demo
YouTube (7 minutes or less): `https://youtu.be/REPLACE_WITH_YOUR_VIDEO`

## Repository layout
```
linear_regression_model/
  render.yaml
  summative/
    pyproject.toml
    uv.lock
    linear_regression/
      multivariate.ipynb
      data/ds_salaries.csv
    API/
      prediction.py
      ml.py
      requirements.txt
      model/
    FlutterApp/
```

## How to run

### Notebook
```bash
cd summative
uv sync --extra notebook
uv run jupyter nbconvert --to notebook --execute --inplace linear_regression/multivariate.ipynb
```

### API locally
```bash
cd summative/API
uv run --project .. uvicorn prediction:app --reload --port 8000
```
Open http://localhost:8000/docs. To retrain and optionally add new rows:
```bash
curl -X POST http://localhost:8000/retrain -F "file=@new_rows.csv"
```

### Deploy the API to Render
1. Push this repo to GitHub.
2. On Render choose New then Blueprint (it reads `render.yaml`), or set up a Web Service with:
   root directory `summative/API`, build command `pip install -r requirements.txt`,
   start command `uvicorn prediction:app --host 0.0.0.0 --port $PORT`.
3. Copy the public URL and update it in the Flutter app and in this README.

### Mobile app
```bash
cd summative/FlutterApp
flutter pub get
```
Open `lib/main.dart` and set `kApiBaseUrl` to your deployed Render URL. If you run the API locally
with an Android emulator, use `http://10.0.2.2:8000`. Then:
```bash
flutter run
```
Fill in the 7 fields, tap Predict, and the app shows the predicted salary or a clear error message
when a value is missing or out of range.

## CORS configuration
The API uses CORS middleware with specific values instead of a wildcard.
- Origins: only our own frontends (the deployed API host, the Flutter web dev server, and localhost).
  A random website cannot call the API from a user's browser. Native mobile apps are not browsers,
  so CORS does not affect them and they still work.
- Methods: only GET, POST and OPTIONS, which are the verbs the API uses.
- Headers: only Content-Type, which is all this JSON API needs.
- Credentials: turned off, because the API is stateless with no cookies or sessions.

## Data types and ranges
The `SalaryRequest` model enforces the types and ranges: `work_year` is an integer between 2020 and
2027, `remote_ratio` is one of 0, 50 or 100, and the other fields are fixed sets of string values.
Any bad type or out of range value returns HTTP 422.
