# Extract the narrative part from the current CODE.md
awk '/# Codebase/{exit} {print}' CODE.md > narrative.md

echo "# Codebase" > code_only.md
echo "" >> code_only.md

# Loop through all py files in babyvllm and tests
for file in $(find babyvllm tests -name "*.py" | sort); do
    echo "## $file" >> code_only.md
    echo "\`\`\`python" >> code_only.md
    cat $file >> code_only.md
    echo "\`\`\`" >> code_only.md
    echo "" >> code_only.md
done

cat narrative.md code_only.md > CODE.md
rm narrative.md code_only.md
