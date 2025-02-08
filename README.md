# California WARN Notice Dashboard

This dashboard analyzes [Worker Adjustment and Retraining Notification (WARN) notices](https://www.dol.gov/general/topic/termination/plantclosings) 
in [California](https://edd.ca.gov/en/jobs_and_training/Layoff_Services_WARN/) for the current year. WARN notices are required when companies 
plan significant layoffs, providing advance notification to affected workers. The notice includes information on the number of affected employees,
notice and effective dates, layoff reasons, and company details.

## Data

The data is being downloaded nightly from the [California Employment Development Department (EDD)'s WARN Page](https://edd.ca.gov/en/jobs_and_training/Layoff_Services_WARN/), converted from XLSX to CSV and Parquet, and stored in a [separate repository](https://github.com/coatless-data/warn-files).

The data includes the following columns:

- `notice_date`: Filing date
- `effective_date`: Layoff/closure date
- `company`: Company name
- `layoff_closure`: Event type
- `no_of_employees`: Affected employees count
- `related_industry`: Industry sector
- `county_parish`: Location
- `state`: Name of the State
- `download_date`: Date of download

You can find the data in the [`data/processed/ca/{YEAR}/`](https://github.com/coatless-data/warn-files/tree/main/data/processed/ca/) directory of the separate repository.

## Setup

To view the dashboard locally, you will need to install the dependencies and render the dashboard.

First, clone the repository:

```bash
git clone git@github.com:coatless-dashboard/warn.git
```

Then, run the following commands:

```bash
# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download data
python .github/scripts/download-warn.py warn-data/data

# Create dashboard
quarto render
```

## Data Refreshes

The data is stored in a [separate repository](https://github.com/coatless-data/warn-files) to allow for easy access, sharing, and versioning.

We download the data nightly from EDD using a GitHub Actions workflow. The workflow is defined in the [`.github/workflows/refresh-warn-data.yml`](.github/workflows/refresh-warn-data.yml). The workflow is triggered daily at 12:00 AM UTC by a CRON job.

## License

AGPL (>= 3)

---

Built with [Quarto](https://quarto.org) and Python. Data from [California Employment Development Department](https://edd.ca.gov/).