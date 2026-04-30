import os
import pandas as pd
from pathlib import Path


def find_xlsx_files(root_dir):
    xlsx_files = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.xlsx'):
                full_path = os.path.join(dirpath, filename)
                xlsx_files[filename] = full_path
    return xlsx_files


def parse_centers(value):
    if pd.isna(value) or value is None or str(value).strip() == '' or str(value) == 'NaN':
        return 0
    return len(str(value).split(';'))


def compare_xlsx_files(template_path, output_path, filename):
    template_df = pd.read_excel(template_path, header=0)
    output_df = pd.read_excel(output_path, header=0)

    template_keys = set(template_df['matched_key'].dropna().astype(str))
    output_keys = set(output_df['matched_key'].dropna().astype(str))

    common_keys = template_keys & output_keys
    missing_in_output = template_keys - output_keys

    if missing_in_output:
        print(f"[DIFF] {filename} - Missing matched_keys in output ({len(missing_in_output)}):")
        for key in sorted(missing_in_output):
            t_row = template_df[template_df['matched_key'].astype(str) == key].iloc[0]
            t_fn = str(t_row['filename']).strip() if 'filename' in t_row and not pd.isna(t_row['filename']) else ''
            print(f"  - {key} ({t_fn})")
        print()

    fields_to_compare = [
        'template_match_res','color_white',
        'red_centers', 'yellow_centers', 'green_centers',"blue_centers"
    ]

    all_match = True
    for key in sorted(common_keys):
        t_row = template_df[template_df['matched_key'].astype(str) == key].iloc[0]
        o_row = output_df[output_df['matched_key'].astype(str) == key].iloc[0]

        diffs = []
        for field in fields_to_compare:
            t_val = t_row[field]
            o_val = o_row[field]

            # if field in ['red_centers', 'yellow_centers', 'green_centers']:
            #     t_count = parse_centers(t_val)
            #     o_count = parse_centers(o_val)
            #     if t_count != o_count:
            #         diffs.append(f"  [{field}] template={t_count}, output={o_count}")
            # else:
            t_str = str(t_val).strip() if not pd.isna(t_val) else ''
            o_str = str(o_val).strip() if not pd.isna(o_val) else ''
            if t_str != o_str:
                try:
                    t_num = float(t_str) if t_str.replace('.', '').replace('-', '').isdigit() else None
                    o_num = float(o_str) if o_str.replace('.', '').replace('-', '').isdigit() else None
                    if t_num is not None and o_num is not None and t_num == o_num:
                        continue
                except:
                    pass
                diffs.append(f"  [{field}] template='{t_str}', output='{o_str}'")

        if diffs:
            all_match = False
            t_filename = str(t_row['filename']).strip() if 'filename' in t_row and not pd.isna(t_row['filename']) else ''
            o_filename = str(o_row['filename']).strip() if 'filename' in o_row and not pd.isna(o_row['filename']) else ''
            print(f"[DIFF] {filename} - template: {t_filename}, output: {o_filename} - {key}:")
            for d in diffs:
                print(d)

    if all_match:
        print("[OK] All fields are consistent.")
    else:
        print(f"\n[INFO] Found differences in above fields.")

def mainc():
    template_dir = r"./template"
    output_dir = r"./output"

    template_files = find_xlsx_files(template_dir)
    output_files = find_xlsx_files(output_dir)

    print(f"Template xlsx: {list(template_files.keys())}")
    print(f"Output xlsx:   {list(output_files.keys())}")

    for filename in template_files:
        if filename in output_files:
            print(f"\n{'='*60}")
            print(f"Comparing: {filename}")
            print(f"Template: {template_files[filename]}")
            print(f"Output:   {output_files[filename]}")
            compare_xlsx_files(template_files[filename], output_files[filename], filename)
        else:
            print(f"\n[MISSING] {filename} not found in output folder.")

if __name__ == "__main__":
    mainc()