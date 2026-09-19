#!/bin/bash
# Copy the exported models (ios/models) into the demo app's Documents/models on a connected device.
# The app must be installed first. Models can also be dropped into the app's folder with the Files app or Finder.
#
#   ios/push_models.sh <device-id> [bundle-id] [models-dir]        device ids: xcrun devicectl list devices
set -eu
DEVICE=${1:?usage: push_models.sh <device-id> [bundle-id] [models-dir]}
BUNDLE=${2:-com.example.PromptMoGeDemo}
MODELS=${3:-"$(cd "$(dirname "$0")" && pwd)/models"}
for d in vit A B; do
  [ -d "$MODELS/$d" ] || { echo "skip $d (not in $MODELS)"; continue; }
  xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE" \
    --source "$MODELS/$d" --destination "Documents/models/$d" > /dev/null
  echo "pushed $d ($(du -sh "$MODELS/$d" | cut -f1))"
done
