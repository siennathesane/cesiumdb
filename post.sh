#!/bin/sh

set -eu

echo "getting diff"

prior_tag=$(git describe --tags --abbrev=0 "$(git rev-list --tags --skip=1 --max-count=1)")
current_tag=$(git describe --tags --abbrev=0)

git diff --patch $current_tag $prior_tag > patchf
grep '^diff --git' patchf | awk '{print $3}' | sed 's|a/||' > filesf

result=()

while IFS= read -r file_path; do
    # check if the file exists
    if [ -f "$file_path" ]; then
        # if the file exists, add it to the result array
        result+=("-f" "$file_path")
    fi
done < filesf

echo "making claude give me patch notes"

changes=$(aichat "${result[@]}" "generate a bullet list of changes to the database code for developers of all skill levels and backgrounds that can be used in release notes. copy this prompt's grammatical structure (re: no caps). use markdown.")

echo "running benchmarks"

bench=$(cargo bench)

echo "replacing variables"

cat << EOF > post.md
# CesiumDB Benchmark Update

## System Info

\`\`\`
$(./sysinfo.py)
\`\`\`

## Benchmark Changes ($(git branch --show-current) ${prior_tag} -> ${current_tag})

${changes}

# Results

to read the benchmark groups, the structure is like this:

- operation/size of value for non-batched operations
- operation/batch/size of batch/size of value for batched operations

\`\`\`
${bench}
\`\`\`
EOF

rm patchf filesf
