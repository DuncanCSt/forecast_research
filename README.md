# forecast_research
Repo to hold research related files and code scripts.

## Cloning the repo
This repo uses a custom branch of hmmTMB which requires cloning both this repo and duncanCSt/hmmTMB as a submodules:

```bash
git clone --recurse-submodules <this-repo-url>
# or for an existing clone
git submodule update --init --recursive
```

## Running with Docker
This repo uses a mix of python and R files, the dependencie structure is complicated so it should only be run in a docker container.
Ensure you have Docker installed and running. Use ```docker info``` to test you have it installed and running.

Build the image (only required the first time, or if dependencies change):

```bash
docker build -t forecast-research:latest 
```

Launch a container with the repository mounted so changes are picked up automatically, and port 8888 forwarded for jupyter notebooks.

```bash
docker run --rm -it -p 8888:8888 -v "$PWD":/app -w /app forecast-research:latest bash
```

Within the container you can run scripts
```bash
python test_wrapper.py
```

or start Jupyter Lab:

```bash
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```
