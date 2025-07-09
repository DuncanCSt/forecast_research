FROM mambaorg/micromamba:1.5.1

# Copy environment definition
COPY environment.yml /tmp/environment.yml

# Create the conda environment
RUN micromamba env create -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Activate the environment by default
ENV PATH="/opt/conda/envs/ts-forecast/bin:$PATH"
SHELL ["micromamba", "run", "-n", "ts-forecast", "--"]

# Workdir where the repository will be mounted
WORKDIR /app

CMD ["bash"]
