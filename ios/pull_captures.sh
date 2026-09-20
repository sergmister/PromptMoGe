#!/bin/bash
# Copy every saved capture (Documents/captures) from the demo app on a connected device.
#
#   ios/pull_captures.sh <device-id> [bundle-id] [destination]       device ids: xcrun devicectl list devices
set -eu
DEVICE=${1:?usage: pull_captures.sh <device-id> [bundle-id] [destination]}
BUNDLE=${2:-com.example.PromptMoGeDemo}
DEST=${3:-captures}
mkdir -p "$DEST"
xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE" \
  --source Documents/captures --destination "$DEST" > /dev/null
echo "$(find "$DEST" -name meta.json | wc -l | tr -d ' ') captures in $DEST  (read them with: python ios/read_capture.py <capture-dir>)"
