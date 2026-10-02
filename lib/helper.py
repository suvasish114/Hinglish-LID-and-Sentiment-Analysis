import pandas as pd

# this function add unique id to the rows of CSV file for better accessability
def add_unique_id(input_file, output_file):
    print("[+] verifying CSV file")
    df = pd.read_csv(input_file)
    if "id" in df.columns: # sanity check
        print(f"[-] 'id' column already exists in {input_file}. No changes made.")
        return
    print("[+] adding unique id column to CSV file ...")
    df.insert(0, "id", range(1, len(df) + 1))
    df.to_csv(output_file, index=False)
    print(f"[+] saved {len(df)} rows to: {output_file}")
