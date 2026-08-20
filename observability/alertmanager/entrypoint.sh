#!/bin/sh
# Render ALERTMANAGER_WEBHOOK_URL into the Alertmanager config and exec it.
#
# Alertmanager cannot substitute environment variables in its config file, so
# the committed alertmanager.yml carries a __ALERTMANAGER_WEBHOOK_URL__
# placeholder; this entrypoint replaces it with the compose-provided URL (which
# defaults to a dead loopback port for UI-only alerting, issue #292) before
# launching the real binary.
set -eu

sed "s|__ALERTMANAGER_WEBHOOK_URL__|${ALERTMANAGER_WEBHOOK_URL}|g" \
    /etc/alertmanager/alertmanager.yml > /tmp/alertmanager.yml

exec alertmanager --config.file=/tmp/alertmanager.yml --storage.path=/alertmanager
