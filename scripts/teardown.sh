#!/usr/bin/env bash
# Delete deployments so GPU billing stops.   scripts/teardown.sh <deployment-id> [...]
set -euo pipefail
firectl deployment list
for id in "$@"; do
  firectl deployment delete "$id" --wait
done
echo "Remaining deployments:"
firectl deployment list
