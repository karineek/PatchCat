#!/usr/bin/env bash
# Current directory (absolute path)
mypath="$(pwd)"

# Usage: ./replace_patchcat_path.sh path/to/file
file="$mypath/src/main/java/gin/util/LocalSearchSimple.java"

# The new line we want to force in, regardless of what was there before
replacement="                    \"$mypath/clustering/PatchCat/PatchCat.py\", // REPLACEMEVIASCRIPT"

# Create a temp file, do the replacement there, then move it back
tmpfile="${file}.tmp"

# Replace any line that ends with: PatchCat.py", // REPLACEMEVIASCRIPT
# No assumptions on what comes before it.
sed -E "s|.*PatchCat\.py\", // REPLACEMEVIASCRIPT|$replacement|" "$file" > "$tmpfile"

mv "$tmpfile" "$file"
