#!/usr/bin/env bash
# Point the AWS CLI and SDKs on your machine at a running emulator.
#
#   eval "$(infra/scripts/aws-env.sh primary)"   # us-east-1, port 9297
#   eval "$(infra/scripts/aws-env.sh dr)"        # us-west-2, port 9298
#   aws cloudformation describe-stacks --query "Stacks[].[StackName,StackStatus]"
#
# Copies the credentials and CA certificate the emulator wrote (they change on every start)
# to infra/.emulator/<which>/ and prints the environment variables to use them.
set -euo pipefail
which="${1:-primary}"
case "$which" in
  primary) container=beyondrag_aws_primary; port=9297; region=us-east-1 ;;
  dr)      container=beyondrag_aws_dr;      port=9298; region=us-west-2 ;;
  *) echo "usage: $0 primary|dr" >&2; exit 2 ;;
esac
here="$(cd "$(dirname "$0")/.." && pwd)"
dest="$here/.emulator/$which"
mkdir -p "$dest"
docker cp "$container:/home/emulator/.aws/emulator/$port/emulator_credentials" "$dest/credentials" >/dev/null
docker cp "$container:/home/emulator/.aws/emulator/$port/ca.pem" "$dest/ca.pem" >/dev/null
cat <<EOF
export AWS_ENDPOINT_URL=https://127.0.0.1:$port
export AWS_SHARED_CREDENTIALS_FILE=$dest/credentials
export AWS_CA_BUNDLE=$dest/ca.pem
export AWS_REGION=$region AWS_DEFAULT_REGION=$region
export AWS_EC2_METADATA_DISABLED=true AWS_DISABLE_HOST_PREFIX_INJECTION=true
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
EOF
