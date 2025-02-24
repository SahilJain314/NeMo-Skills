import json
import sys

# Define input and output file paths
input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Parse JSON from the current line
        sample = json.loads(line)
        
        # Add the "task_name": "math" key-value pair
        sample['task_name'] = 'math'
        
        # Write the updated sample to the output file
        outfile.write(json.dumps(sample) + '\n')
