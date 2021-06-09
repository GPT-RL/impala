FROM python:3.8
RUN pip3 install bsuite && \
    python3.8 -c \
    "from bsuite.sweep import SWEEP; print('RPUSH env-queue ' + ' '.join(SWEEP), end='')" \
    > /envs.redis

FROM redis:alpine
WORKDIR /workspace
COPY --from=0 /envs.redis .
COPY run.sh .

ENTRYPOINT ["/workspace/run.sh"]