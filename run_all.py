import os
import subprocess

def run_script(script_path):
    print(f"\n>>> Running: {script_path}")
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

def main():
    print("STARTING COUNTERFACTUAL AFRICA DEMO")
    
    # Create outputs folder if doesn't exist
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    
    run_script('scripts/process_data.py')
    run_script('scripts/run_demo.py')
    
    print("\n*** DEMO COMPLETED SUCCESSFULLY! ***")

if __name__ == "__main__":
    main()
