#!/bin/sh
# Save a post worth keeping (usually from X) to the Bookmarks page, with your take.
#
#   sh scripts/bookmark.sh https://x.com/<handle>/status/<id>
#
# Asks for the author, the passage (paste it, then a line with just "."), your take
# (same) and the tags. Writes content/bookmarks/<date>-<handle>-<id>.md. Nothing is
# fetched from X: you paste what you want to keep. Then commit and push:
#   git add content/bookmarks data/tags.yaml && git commit -m "Bookmark: ..." && git push
set -eu
cd "$(dirname "$0")/.."
url=${1:?usage: sh scripts/bookmark.sh <link to the post>}

handle=$(printf '%s' "$url" | sed -nE 's#^https?://(www\.)?(x|twitter)\.com/([A-Za-z0-9_]+)/status/.*#\3#p')
id=$(printf '%s' "$url" | sed -nE 's#.*/status/([0-9]+).*#\1#p')
day=$(date +%Y-%m-%d)
slug=${handle:-link}-${id:-$(date +%H%M%S)}
file=content/bookmarks/$day-$slug.md
[ -e "$file" ] && { echo "already saved: $file"; exit 1; }

# Reads lines until one that is just "."; prints them.
block() { while IFS= read -r line; do [ "$line" = "." ] && break; printf '%s\n' "$line"; done; }

printf 'Author name%s: ' "${handle:+ (@$handle)}"; IFS= read -r author
[ -n "$handle" ] || { printf 'Handle (without @, optional): '; IFS= read -r handle; }
echo 'The passage worth keeping (a few lines, not the whole thread). End with a line ".":'
quote=$(block)
echo 'Your take: what it changed, where you disagree, what you will try. End with ".":'
take=$(block)

# Tags come from the fixed vocabulary in data/tags.yaml (the build stops on any other
# spelling). What you type is matched to it whatever its case, so "rag" is saved as
# "RAG". For a new tag, add a line to that file, then type the tags again.
vocab=data/tags.yaml
known() { sed -n 's/^- *//p' "$vocab"; }
while :; do
  printf 'Tags, comma-separated (optional), from: %s\n> ' "$(known | paste -sd, - | sed 's/,/, /g')"
  IFS= read -r typed || typed=
  tags=$(printf '%s\n' "$typed" | tr ',' '\n' | sed 's/^ *//;s/ *$//;/^$/d' | while IFS= read -r t; do
    known | grep -ixF -- "$t" || { echo "not in $vocab: $t" >&2; exit 1; }
  done) && break
done

[ -n "$quote" ] || { echo "no passage, nothing saved"; exit 1; }
[ -n "$take" ] || { echo "no take, nothing saved: the take is the point"; exit 1; }

sq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/''/g")"; }
{
  echo '---'
  echo "date: $(date +%Y-%m-%dT%H:%M:%S%z | sed 's/\(..\)$/:\1/')"
  echo "source: $(sq "$url")"
  echo "author: $(sq "$author")"
  echo "handle: $(sq "$handle")"
  echo 'quote: |'
  printf '%s\n' "$quote" | sed 's/^/  /'
  printf 'tags: ['
  printf '%s\n' "$tags" | sed '/^$/d' | sed "s/'/''/g;s/.*/'&'/" | paste -sd, - | tr -d '\n'
  echo ']'
  echo '---'
  echo
  printf '%s\n' "$take"
} > "$file"
echo "saved $file"
