import os
import time
import subprocess

def read_run_all(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    parameters_list = content.split('parameter:')
    parameters_list = [param.strip() for param in parameters_list if param.strip()]
    parsed_parameters = []
    
    for param in parameters_list:
        param_dict = {}
        lines = param.split('\n')
        for line in lines:
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                param_dict[key.strip()] = value.strip()
        parsed_parameters.append(param_dict)
    
    return parsed_parameters

def write_parameters_file(parameters, file_path='parameters.txt'):
    with open(file_path, 'w') as file:
        for key, value in parameters.items():
            file.write(f"{key} = {value}\n")

def run_gsa_script():
    subprocess.run(['python', 'gsa.py', '--auto'])

def main():
    run_all_path = 'run_all.txt'
    parameters_list = read_run_all(run_all_path)
    completion_file = 'calculation_completed.txt'
    
    for i, parameters in enumerate(parameters_list, start=1):
        # Write parameters to parameters.txt
        write_parameters_file(parameters)
        
        # Remove the completion file if it exists
        if os.path.exists(completion_file):
            os.remove(completion_file)
        
        # Run the gsa.py script
        run_gsa_script()
        
        # Wait for the completion file to appear
        while not os.path.exists(completion_file):
            time.sleep(1)
        
        print(f"Processed parameter set {i}:")
        for key, value in parameters.items():
            print(f"{key} = {value}")
        print("\n")

if __name__ == "__main__":
    main()
