
input_file_path = '/ivy_models/ivy_models/googlenet/googlenet.py'
output_file_path = 'm_' + input_file_path 

def remove_lines_starting_with(prefix, input_file, output_file):
    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            if not line.strip().startswith(prefix):
                output_f.write(line)


remove_lines_starting_with(prefix_to_remove, input_file_path, output_file_path)
print("Lines starting with '{}' removed.".format(prefix_to_remove))