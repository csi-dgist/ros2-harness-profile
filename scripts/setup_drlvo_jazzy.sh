#!/usr/bin/env bash
set -eo pipefail

readonly root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ws="${DRLVO_WS:-$HOME/drl_vo_jazzy_ws}"
readonly repo="${ws}/src/drl_vo_nav"
readonly result="${root}/results/drlvo_setup_v1"
readonly branch="humble"
readonly expected_commit="93b771a8c365781af2df479557d45f9a76f68756"

mkdir -p "${ws}/src" "${result}"
printf 'setup_started=%s\n' "$(date --iso-8601=seconds)" >> "${result}/progress.log"

if [[ ! -d "${repo}/.git" ]]; then
  git clone --branch "${branch}" --single-branch \
    https://github.com/TempleRAIL/drl_vo_nav.git "${repo}" \
    >> "${result}/build.log" 2>&1
fi

git -C "${repo}" fetch origin "${branch}" >> "${result}/build.log" 2>&1
git -C "${repo}" checkout --detach "${expected_commit}" >> "${result}/build.log" 2>&1
actual_commit="$(git -C "${repo}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  printf 'commit_mismatch expected=%s actual=%s\n' "${expected_commit}" "${actual_commit}" >> "${result}/progress.log"
  exit 2
fi
if git -C "${repo}" diff --quiet; then
  clean_upstream_source=1
else
  clean_upstream_source=0
  printf 'preexisting_compatibility_diff=%s\n' "$(date --iso-8601=seconds)" >> "${result}/progress.log"
  git -C "${repo}" diff --binary > "${result}/preexisting_compatibility.patch"
fi

source /opt/ros/jazzy/setup.bash
set -u
sudo -n apt-get update >> "${result}/build.log" 2>&1
sudo -n apt-get install -y \
  python3-colcon-common-extensions python3-rosdep python3-venv \
  ros-jazzy-cv-bridge ros-jazzy-image-transport \
  >> "${result}/build.log" 2>&1

readonly uv_bootstrap="${ws}/uv-bootstrap"
if [[ ! -x "${uv_bootstrap}/bin/uv" ]]; then
  python3 -m venv "${uv_bootstrap}"
  "${uv_bootstrap}/bin/python" -m pip install --disable-pip-version-check "uv==0.8.11" \
    >> "${result}/build.log" 2>&1
fi
export PATH="${uv_bootstrap}/bin:${PATH}"
uv --version > "${result}/uv_version.txt"

cd "${ws}"
colcon list --base-paths "${repo}/nav2py_drl_vo" \
  > "${result}/colcon_packages.txt"
set +e
colcon build \
  --base-paths "${repo}/nav2py_drl_vo" \
  --packages-up-to nav2py_drl_vo_controller \
  --symlink-install \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  >> "${result}/build.log" 2>&1
build_status=$?
set -e

if [[ ${build_status} -ne 0 && ${clean_upstream_source} -eq 1 ]]; then
  printf 'unmodified_jazzy_build_failed=%s status=%s\n' \
    "$(date --iso-8601=seconds)" "${build_status}" >> "${result}/progress.log"
  cp -f "${result}/build.log" "${result}/unmodified_build.log"
  compatibility_patch="${root}/patches/drlvo_jazzy_std_msgs.patch"
  if git -C "${repo}" apply --check "${compatibility_patch}"; then
    git -C "${repo}" apply "${compatibility_patch}"
    git -C "${repo}" diff --binary > "${result}/applied_compatibility.patch"
    sha256sum "${compatibility_patch}" > "${result}/compatibility_patch.sha256"
    printf 'minimal_build_metadata_patch_applied=%s\n' "$(date --iso-8601=seconds)" >> "${result}/progress.log"
    colcon build \
      --base-paths "${repo}/nav2py_drl_vo" \
      --packages-up-to nav2py_drl_vo_controller \
      --symlink-install \
      --event-handlers console_direct+ \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      >> "${result}/build.log" 2>&1
  else
    printf 'compatibility_patch_not_applicable=%s\n' "$(date --iso-8601=seconds)" >> "${result}/progress.log"
    exit "${build_status}"
  fi
elif [[ ${build_status} -ne 0 ]]; then
  printf 'patched_jazzy_build_failed=%s status=%s\n' \
    "$(date --iso-8601=seconds)" "${build_status}" >> "${result}/progress.log"
  exit "${build_status}"
elif [[ ${clean_upstream_source} -eq 1 ]]; then
  printf 'unmodified_jazzy_build_succeeded=%s\n' "$(date --iso-8601=seconds)" >> "${result}/progress.log"
else
  printf 'patched_jazzy_build_succeeded=%s\n' "$(date --iso-8601=seconds)" >> "${result}/progress.log"
fi

model="${repo}/nav2py_drl_vo/nav2py_drl_vo_controller/model/drl_vo.zip"
test -s "${model}"
sha256sum "${model}" > "${result}/model.sha256"
printf 'upstream_commit=%s\n' "${actual_commit}" > "${result}/upstream.txt"
printf 'setup_completed=%s\n' "$(date --iso-8601=seconds)" >> "${result}/progress.log"
