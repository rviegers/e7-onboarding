#!/bin/bash
set -e

BRANCH="session_one"
REPO="git@github.com:rviegers/e7-onboarding.git"
DATA_SRC="/srv/data/onboarding Epoch VII/data"

usage() {
    echo "Usage: $0 [-b branch]"
    echo "  -b  Branch to check out (default: session_one)"
    exit 1
}

while getopts "b:h" opt; do
    case $opt in
        b) BRANCH="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

cd ~

if [ -d "e7-onboarding" ]; then
    echo "ERROR: ~/e7-onboarding already exists. Remove it first and re-run."
    exit 1
fi

echo "Cloning $REPO (branch: $BRANCH)..."
git clone --branch "$BRANCH" "$REPO"

echo "Copying data..."
cp -r "$DATA_SRC" e7-onboarding/data

echo "Installing dependencies..."
cd e7-onboarding
uv sync

echo "Done. Ready in ~/e7-onboarding"
