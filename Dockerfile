FROM mambaorg/micromamba:1.5.1

# Copy environment definition
COPY environment.yml /tmp/environment.yml

# Create the conda environment
RUN micromamba env create -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Activate the environment by default and route RUN via bash
ENV PATH="/opt/conda/envs/ts-forecast/bin:$PATH"
SHELL ["micromamba", "run", "-n", "ts-forecast", "bash", "-lc"]

# Install hmmTMB from vendored submodule (if present)
# Note: requires `forecast_research/vendor/hmmTMB` to exist in build context
COPY vendor/hmmTMB /tmp/hmmTMB
RUN ["micromamba","run","-n","ts-forecast","R","-q","-e","remotes::install_local('/tmp/hmmTMB', dependencies=TRUE)"]

# Workdir where the repository will be mounted
WORKDIR /app

CMD ["bash"]
