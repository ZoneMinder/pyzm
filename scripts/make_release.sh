#!/bin/bash
set -e

SKIP_PYPI=false
for arg in "$@"; do
    case "$arg" in
        --skip-pypi) SKIP_PYPI=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Always target the pliablepixels fork, never upstream
GH_REPO="pliablepixels/pyzm"

# --- Read version from pyzm/__init__.py ---
INIT_FILE="pyzm/__init__.py"
if [ ! -f "$INIT_FILE" ]; then
    echo "ERROR: $INIT_FILE not found"
    exit 1
fi
VER=$(grep -Po '(?<=^__version__ = ["\x27])[^"\x27]+' "$INIT_FILE")
if [ -z "$VER" ]; then
    echo "ERROR: could not parse __version__ from $INIT_FILE"
    exit 1
fi

echo "=== Release v${VER} ==="
echo

# --- Preflight checks ---
for cmd in git-cliff gh; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found."
        exit 1
    fi
done
if [ "$SKIP_PYPI" = false ]; then
    if ! python3 -m build --help &>/dev/null; then
        echo "ERROR: python3 -m build not available. Install with: pip install build"
        exit 1
    fi
    if ! command -v twine &>/dev/null; then
        echo "ERROR: twine not found. Install with: pip install twine"
        exit 1
    fi
fi
export GITHUB_TOKEN=$(gh auth token)

# --- Step 1: Check if tag already exists ---
if git rev-parse "v${VER}" &>/dev/null; then
    echo "Tag v${VER} already exists."
    read -p "Overwrite existing release? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo "  Deleting old release and tag v${VER} ..."
        gh release delete "v${VER}" --repo "$GH_REPO" --yes 2>/dev/null || true
        git tag -d "v${VER}"
        git push origin --delete "v${VER}" 2>/dev/null || true
    else
        echo "Aborted."
        exit 0
    fi
    echo
fi

# --- Step 2: Check for uncommitted files ---
DIRTY_FILES=$(git status --porcelain)
if [ -n "$DIRTY_FILES" ]; then
    # Allow only pyzm/__init__.py (version bump) to be dirty
    NON_INIT=$(echo "$DIRTY_FILES" | grep -v " ${INIT_FILE}$" || true)
    if [ -n "$NON_INIT" ]; then
        echo "ERROR: Uncommitted files besides ${INIT_FILE}:"
        echo "$NON_INIT"
        exit 1
    fi
    echo "Committing ${INIT_FILE} (version bump) ..."
    git add "$INIT_FILE"
    git commit -m "chore: bump version to v${VER}"
    git push origin master
    echo "  Done."
    echo
fi

# --- Confirm before proceeding ---
BRANCH=$(git rev-parse --abbrev-ref HEAD)
REMOTE_URL=$(git remote get-url origin)
echo "--- Release summary ---"
echo "  Version:      v${VER}"
echo "  Branch:       ${BRANCH}"
echo "  Remote:       ${REMOTE_URL}"
echo "  GitHub repo:  ${GH_REPO}"
echo "  PyPI upload:  $([ "$SKIP_PYPI" = true ] && echo "SKIPPED" || echo "yes")"
echo
if [ "$SKIP_PYPI" = true ]; then
    echo "This will: generate CHANGELOG, commit, tag, push, and create GitHub release (PyPI skipped)."
else
    echo "This will: generate CHANGELOG, commit, tag, push, build & upload to PyPI, and create GitHub release."
fi
read -p "Proceed? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi
echo

# --- Step 3: Generate and commit changelog ---
echo "Generating CHANGELOG.md ..."
git-cliff --tag "v${VER}" -o CHANGELOG.md
echo "  Done."

echo "Committing CHANGELOG.md ..."
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for v${VER}"
git push origin master
echo "  Done."
echo

# --- Step 4: Tag ---
echo "Creating tag v${VER} ..."
git tag -a "v${VER}" -m "v${VER}"
git push origin --tags
echo "  Done."
echo

# --- Step 5: Build and upload to PyPI ---
if [ "$SKIP_PYPI" = false ]; then
    echo "Building PyPI packages ..."
    rm -rf dist
    python3 -m build
    echo "  Done."

    echo "Uploading to PyPI ..."
    twine upload dist/pyzm-"${VER}"* --verbose
    echo "  Done."
    echo
else
    echo "Skipping PyPI build & upload (--skip-pypi)."
    echo
fi

# --- Step 6: Create GitHub Release ---
echo "Creating GitHub Release v${VER} ..."
NOTES_FILE=$(mktemp)
git-cliff --latest --strip header > "$NOTES_FILE" 2>/dev/null
gh release create "v${VER}" --repo "$GH_REPO" --title "v${VER}" --notes-file "$NOTES_FILE"
rm -f "$NOTES_FILE"
echo "  Done."

echo
echo "=== Release v${VER} complete ==="
