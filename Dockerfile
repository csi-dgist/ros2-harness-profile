# Clean-environment runner for ae_verify.sh. No ROS 2 and no simulator, so README
# section 2 (one matched pair in Gazebo) is out of scope for this image.
#
#   docker build -t harness-profile-ae .
#   docker run --rm harness-profile-ae
#   docker run --rm -v "$PWD/out:/artifact/out" harness-profile-ae   # keep the outputs

FROM python:3.12-slim

WORKDIR /artifact
COPY requirements-analysis.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements-analysis.txt
COPY . .

ENV SKIP_VENV=1 MPLBACKEND=Agg
CMD ["bash", "ae_verify.sh"]
