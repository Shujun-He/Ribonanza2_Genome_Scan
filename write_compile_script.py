from glob import glob
import os

# Create a directory called 'scripts' if it doesn't already exist
if not os.path.exists('scripts'):
    os.mkdir('scripts')

# Loop through all files in the 'predictions' folder
os.system(f"python compile_parallel.py --chromosome lcl*")
# for f in glob("predictions/*"):
# 	if "CM" in f:

# 		# Get the filename (last part of the path)
# 		filename = f.split("/")[-1]
		
# 		# Create a new script file inside the 'scripts' directory for each file in predictions
# 		script_path = os.path.join('scripts', f"{filename}.sh")
		
# 		with open(script_path, 'w') as script_file:
# 			script_file.write("#!/bin/bash\n")
# 			script_file.write("#SBATCH -p owners\n")
# 			script_file.write("#SBATCH -c 4\n")
# 			script_file.write("#SBATCH --mem=8GB\n")
# 			script_file.write("#SBATCH --time=1:00:00\n")
# 			script_file.write("\n")
# 			script_file.write(f"/scratch/groups/rhiju/shujun/miniconda3/envs/torch/bin/python compile_parallel.py --chromosome {filename}\n")
			
# 		# Print the script path to confirm it's created
# 		print(f"Created script: {script_path}")

# 		#os.system(f"sbatch {script_path}")
# 		os.system(f"python compile_parallel.py --chromosome {filename}")

