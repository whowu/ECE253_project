import os
from step1_baseline import run_step1
from step2_optimize import run_step2
from step3_finetune import run_step3

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("datasets/temp_processed", exist_ok=True)
    
    # Step 1
    run_step1()
    
    # Step 2
    best_configs = run_step2()
    
    # Step 3
    run_step3(best_configs)
    
    print("\n✅ Project Pipeline Complete. Check 'results/' folder.")

if __name__ == "__main__":
    main()