# pf for testing the unit-test itself. 

from pprint import pprint as pp
# setting paths
file_count_path = "/ivy_models/ivy_models/inceptionnetv3/log_sys/file_count.txt"

# Read the current file count from a count file
try:
    with open(file_count_path, "r") as count_file:
        file_count = int(count_file.read())
except FileNotFoundError:
    file_count = 100
# setting more paths
file_backup_path = f'/ivy_models/ivy_models/inceptionnetv3/log_sys/inputs/input_{file_count}.json'
file_latest_path = f'/ivy_models/ivy_models/inceptionnetv3/log_sys/inputs/input.json'

# create outputstream as input_count.json and input.json
file_backup = open(file_backup_path, 'a')
file_latest = open(file_latest_path, 'a')
# increment file count and save to file
file_count -= 1
with open(file_count_path, "w") as count_file:
    count_file.write(str(file_count))

def pf(input):
    # print to input_count.json and input.json
    pp(input, stream=file_backup)
    pp(input, stream=file_latest)
    
# test pf lines
# pf("sarvesh .dsf.")
# pf("sarvesh .dsf234.")
# pf("sarvesh .dsf345389889.")
# pf("sarvesh .dsf.sdsf")


