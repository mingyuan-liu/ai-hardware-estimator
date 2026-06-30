#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${ENV_FILE:-/etc/ai-hardware-estimator-deploy.env}
LOCK_FILE=${LOCK_FILE:-/run/ai-hardware-estimator-deploy.lock}
LOG_DIR=${LOG_DIR:-/var/log/ai-hardware-estimator-deploy}

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

APP_DIR=${APP_DIR:-/srv/ai-hardware-estimator}
APP_BRANCH=${APP_BRANCH:-main}
APP_SERVICE=${APP_SERVICE:-ai-hardware-estimator.service}
FORCE=${FORCE:-0}

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || {
  echo "Another ai-hardware-estimator deploy is running, exit."
  exit 0
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
  log "$*"
  exit 1
}

if [[ ! -d "$APP_DIR/.git" ]]; then
  fail "app is not a git repository: $APP_DIR"
fi

main() {
  local before
  local after
  local changed=0

  if ! before=$(git -C "$APP_DIR" rev-parse HEAD 2>/dev/null); then
    fail "cannot resolve current HEAD in $APP_DIR"
  fi

  if ! git -C "$APP_DIR" fetch --prune origin "$APP_BRANCH"; then
    fail "git fetch origin $APP_BRANCH failed in $APP_DIR"
  fi

  if ! after=$(git -C "$APP_DIR" rev-parse "origin/$APP_BRANCH" 2>/dev/null); then
    fail "cannot resolve origin/$APP_BRANCH in $APP_DIR"
  fi

  if [[ "$before" != "$after" ]]; then
    log "app changed: $before -> $after"
    git -C "$APP_DIR" reset --hard "origin/$APP_BRANCH"
    changed=1
  else
    log "app unchanged: $before"
  fi

  if [[ "$changed" == 1 || "$FORCE" == 1 ]]; then
    log "install dependencies"
    npm --prefix "$APP_DIR" ci
    log "build app"
    npm --prefix "$APP_DIR" run build
    log "restart $APP_SERVICE"
    systemctl restart "$APP_SERVICE"
    log "deploy done"
  else
    log "no changes, skip deploy"
  fi
}

main "$@" 2>&1 | tee -a "$LOG_DIR/deploy.log"
