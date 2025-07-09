# forecast_research
Repo to hold research related files and code scripts.

## Running with Conda

```bash
conda env create -f environment.yml
conda activate ts-forecast
```
## Running with Docker

Build the image:

```bash
docker build -t forecast-research .
```

Launch a container with the repository mounted so changes are picked up automatically:

```bash
docker run --rm -it -v $(pwd):/app forecast-research
```

Within the container you can run scripts or start Jupyter Lab:

```bash
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```
