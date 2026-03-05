#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Find the script's path
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ "${CLEAN_BUILD:-false}" = "true" ]; then
  rm -rf "${SCRIPT_DIR}/../github_clones"
  docker rmi aircraft-image:latest ground-image:latest simulation-image:latest || true
  docker builder prune -f # If CLEAN_BUILD is "true", rebuild everything from scratch
fi

BUILD_DOCKER=true
if [ "${CLONE_ONLY:-false}" = "true" ]; then
  BUILD_DOCKER=false # If CLONE_ONLY is "true", disable the build steps
fi

# Create a folder (ignored by git) to clone GitHub repos
CLONE_DIR="${SCRIPT_DIR}/../github_clones"
mkdir -p "$CLONE_DIR"

REPOS=( # Format: "URL;BRANCH;LOCAL_DIR_NAME"
  # Simulation image
  "https://github.com/PX4/PX4-Autopilot.git;v1.16.1;PX4-Autopilot"
  "https://github.com/ArduPilot/ardupilot.git;Copter-4.6.3;ardupilot"
  "https://github.com/ArduPilot/ardupilot_gazebo.git;main;ardupilot_gazebo"
  "https://github.com/PX4/flight_review.git;main;flight_review"
  # Ground image
  "https://github.com/mavlink/c_library_v2;master;c_library_v2"
  "https://github.com/mavlink-router/mavlink-router;master;mavlink-router"
  # Aircraft image
  "https://github.com/PX4/px4_msgs.git;release/1.16;px4_msgs"
  "https://github.com/eProsima/Micro-XRCE-DDS-Agent.git;master;Micro-XRCE-DDS-Agent"
  "https://github.com/PRBonn/kiss-icp.git;main;kiss-icp"
)

for repo_info in "${REPOS[@]}"; do
  IFS=';' read -r url branch dir <<< "$repo_info" # Split the string into URL, BRANCH, and DIR
  TARGET_DIR="${CLONE_DIR}/${dir}"
  if [ -d "$TARGET_DIR" ]; then
    cd "$TARGET_DIR"
    BRANCH=$(git branch --show-current)
    TAGS=$(git tag --points-at HEAD)
    echo "There is a clone of ${dir} on branch: ${BRANCH}, tags: [${TAGS}]"
    # The script does not automatically pull changes for already cloned repos (as they should be on fixed tags)
    # This avoids breaking the Docker cache but it requires manually deleting the github_clones folder for branch/tag updates
    # git pull
    # git submodule update --init --recursive --depth 1
    cd "$CLONE_DIR"
  else
    echo "Clone not found, cloning ${dir}..."
    TEMP_DIR="${TARGET_DIR}_temp"     
    rm -rf "$TEMP_DIR" # Clean up any failed clone from a previous run   
    git clone --depth 1 --branch "$branch" --recursive "$url" "$TEMP_DIR" && mv "$TEMP_DIR" "$TARGET_DIR"
  fi
done

# Get simulation_assets from GitHub release
ASSETS_URL="https://github.com/JacopoPan/aerial-autonomy-stack/releases/download/v1.1.0/simulation_assets.zip"
EXPECTED_HASH="906de433a5b9a0b896b30dde1642d103c7ffeac9f71ae8d9b3f95a9847ea153a" # sha256sum simulation_assets.zip
ZIP_FILE="$CLONE_DIR/simulation_assets.zip"
DOWNLOAD_NEEDED=true
if [ -f "$ZIP_FILE" ]; then
  CURRENT_HASH=$(sha256sum "$ZIP_FILE" | awk '{print $1}')
  if [ "$CURRENT_HASH" = "$EXPECTED_HASH" ]; then
    DOWNLOAD_NEEDED=false
  fi
fi
if [ "$DOWNLOAD_NEEDED" = "true" ]; then
  echo "Downloading simulation assets from $ASSETS_URL..."
  wget -q -O "$ZIP_FILE" "$ASSETS_URL"
  DOWNLOAD_HASH=$(sha256sum "$ZIP_FILE" | awk '{print $1}')
  if [ "$DOWNLOAD_HASH" != "$EXPECTED_HASH" ]; then
    echo -e "ERROR: The downloaded file hash is incorrect\nExpected: $EXPECTED_HASH\nGot: $DOWNLOAD_HASH"
    exit 1 # Stop the script
  fi
fi
# Unzip quietly (-q), overwrite (-o), into the repository root directory (-d) above scripts/ to merge into simulation/
unzip -q -o "$ZIP_FILE" -d "$SCRIPT_DIR/.."

if [ "$BUILD_DOCKER" = "true" ]; then
  # The first build takes ~15' and creates a 21GB image (8GB for ros-humble-desktop with nvidia runtime, 10GB for PX4 and ArduPilot SITL)
  docker build -t simulation-image -f "${SCRIPT_DIR}/docker/Dockerfile.simulation" "${SCRIPT_DIR}/.."

  # The first build takes <5' and creates an 9GB image (8GB for ros-humble-desktop with nvidia runtime)
  docker build -t ground-image -f "${SCRIPT_DIR}/docker/Dockerfile.ground" "${SCRIPT_DIR}/.."

  # The first build takes ~10' and creates an 18GB image (8GB for ros-humble-desktop with nvidia runtime, 7GB for YOLO, ONNX)
  docker build -t aircraft-image -f "${SCRIPT_DIR}/docker/Dockerfile.aircraft" "${SCRIPT_DIR}/.."
else
  echo -e "Skipping Docker builds"
fi