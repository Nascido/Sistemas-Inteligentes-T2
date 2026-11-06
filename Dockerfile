ARG JUPYTER_IMAGE=quay.io/jupyter/scipy-notebook:2026-07-28
FROM ${JUPYTER_IMAGE}

COPY --chown=1000:100 requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --requirement /tmp/requirements.txt \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

WORKDIR /home/jovyan/work
