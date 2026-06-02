import pandas as pd
import re
import json

df = pd.read_excel('/Users/saw/WorkSpace/work/OCR-Project/data/位置.xlsx')
print("第一列名称:", df.columns[0])
print(f"共 {len(df)} 行数据\n")

with open('/Users/saw/WorkSpace/work/OCR-Project/resource/target.json', 'r', encoding='utf-8') as f:
    target_data = json.load(f)

target_strings = set()
for key, values in target_data.items():
    target_strings.add(key)
    for value in values:
        target_strings.add(value)

print(f"target.json 中共有 {len(target_strings)} 个字符串\n")

def has_non_ascii(s):
    return any(ord(ch) > 127 for ch in s)

def to_half_width(s):
    result = []
    for ch in s:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(ch)
    return ''.join(result)

res = []
res0 = []
for idx, val in enumerate(df.iloc[:, 0].tolist()):
    if pd.isna(val):
        print(f"{idx}: NaN")
        continue
    val_str = str(val).strip()
    parts = val_str.split("-")
    nums = parts[1].split(" ")
    num = nums[0]
    if num == '4':
        print(0)
    if "D" not in num:
        num = to_half_width(num)
        if num == "S202":
            print(-1)
        if num == '':
            continue
        if '0' in num:
            res0.append(num)
        elif '\\' in num or has_non_ascii(num):
            continue
        else:
            res.append(num)

print("\n去重后（不含0）:")
res = list(set(res))
res = sorted(res)
print(len(res))
print(res)

print("\n含0或\\或中文的结果:")
res0 = list(set(res0))
res0 = sorted(res0)
print(len(res0))
print(res0)

print("\n" + "="*60)
print("在 Excel 中但不在 target.json 中的字符串")
print("="*60)

all_excel_strings = set(res + res0)
not_in_target = []
for s in sorted(all_excel_strings):
    if s not in target_strings:
        not_in_target.append(s)

print(f"\n共有 {len(not_in_target)} 个字符串不在 target.json 中：")
print(not_in_target)
s = ['1FS', '1FSL', '1FSZ1', '1FX1', '1FX3', '1FXF', '1FXL', '2FS', '2FSI', '2FSL', '2FSZ1', '2FXF', '2FXL', '3FXI', '3FXL', 'FS1', 'FS3', 'FSL209', 'FSPN', 'FSZ3', 'FTF2', 'FTF3', 'FTF4', 'FTF5', 'FTF6', 'FTF7', 'FTF8', 'FX', 'FX1', 'FX3', 'FX4', 'FX6', 'FX8', 'FXB', 'FXZ1', 'H', 'S12', 'S1F', 'S3F', 'SF1', 'SF2', 'SFA', 'SFB', 'SFC', 'SJ', 'SK', 'SP', 'SPF', 'SPN', 'SV', 'SVI', 'SY', 'SZ', 'SZ1', 'SZ2', 'SZ3', 'SZ4', 'SZ5', 'T1', 'T1F', 'T2', 'T2F', 'TF2', 'TF3', 'TF4', 'TF5', 'TF6', 'TF7', 'TF8', 'TF9', 'X12', 'X13', 'XB', 'XBN', 'XC', 'XFA', 'XFB', 'XFC', 'XFZ2', 'XFZ3', 'XFZ4', 'XFZ5', 'XL1', 'XL3', 'XV', 'XVI', 'XY', 'XZ1', 'XZ2', 'XZ3', 'XZ3F', 'XZ4', 'XZ5', 'XZ7', 'XZF', 'YS', 'YSF', 'YSK', 'YXB', 'YXF']
print(len(s))
# 去重后（不含0）:
# 179
# ['1FS', '1FSL', '1FSZ1', '1FX1', '1FX3', '1FXF', '1FXL', '2FS', '2FSI', '2FSL', '2FSZ1', '2FXF', '2FXL', '3FXI', '3FXL', 'B1', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B2', 'B21', 'B22', 'B23', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'FS', 'FS1', 'FS3', 'FSI', 'FSL', 'FSL2', 'FSPN', 'FSZ3', 'FTF2', 'FTF3', 'FTF4', 'FTF5', 'FTF6', 'FTF7', 'FTF8', 'FX', 'FX1', 'FX3', 'FX4', 'FX6', 'FX8', 'FXB', 'FXF', 'FXI', 'FXZ1', 'H', 'S', 'S1', 'S11', 'S12', 'S15', 'S1F', 'S2', 'S3', 'S3F', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'SA', 'SC', 'SF', 'SF1', 'SF2', 'SFA', 'SFB', 'SFC', 'SI', 'SII', 'SIV', 'SJ', 'SK', 'SL', 'SL12', 'SL14', 'SL2', 'SL4', 'SL6', 'SL8', 'SN', 'SP', 'SPF', 'SPN', 'SV', 'SVI', 'SY', 'SZ', 'SZ1', 'SZ2', 'SZ3', 'SZ4', 'SZ5', 'T1', 'T1F', 'T2', 'T2F', 'TF2', 'TF3', 'TF4', 'TF5', 'TF6', 'TF7', 'TF8', 'TF9', 'X', 'X1', 'X11', 'X12', 'X13', 'X15', 'X1F', 'X2', 'X3', 'X325', 'X3F', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'XA', 'XB', 'XBN', 'XC', 'XF', 'XFA', 'XFB', 'XFC', 'XFZ2', 'XFZ3', 'XFZ4', 'XFZ5', 'XI', 'XII', 'XIV', 'XL', 'XL1', 'XL3', 'XN', 'XV', 'XVI', 'XY', 'XZ', 'XZ1', 'XZ2', 'XZ3', 'XZ3F', 'XZ4', 'XZ5', 'XZ7', 'XZF', 'YS', 'YSA', 'YSC', 'YSF', 'YSK', 'YX', 'YXA', 'YXB', 'YXF']
#
# 含0或\或中文的结果:
# 40
# ['B10', 'B20', 'FSL209', 'S10', 'S202', 'S204', 'S205', 'S20Ⅲ', 'SL10', 'SL201', 'SL202', 'SL203', 'SL301', 'SL401', 'SL402', 'SL404', 'SL405', 'SL406', 'SL407', 'SL408', 'X10', 'X201', 'X202', 'X203', 'X204', 'X205', 'X206', 'X207', 'X208', 'X209', 'X20Ⅲ', 'X20Ⅳ', 'X401', 'X402', 'X404', 'X405', 'X406', 'X407', 'X408', 'X40Ⅲ']
# 'B1', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B2', 'B20', 'B21', 'B22', 'B23', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9'
# ['1FS', '1FSL', '1FSZ1', '1FX1', '1FX3', '1FXF', '1FXL', '2FS', '2FSI', '2FSL', '2FSZ1', '2FXF', '2FXL', '3FXI', '3FXL', 'FS1', 'FS3', 'FSL209', 'FSPN', 'FSZ3', 'FTF2', 'FTF3', 'FTF4', 'FTF5', 'FTF6', 'FTF7', 'FTF8', 'FX', 'FX1', 'FX3', 'FX4', 'FX6', 'FX8', 'FXB', 'FXZ1', 'H', 'S12', 'S1F', 'S3F', 'SF1', 'SF2', 'SFA', 'SFB', 'SFC', 'SJ', 'SK', 'SP', 'SPF', 'SPN', 'SV', 'SVI', 'SY', 'SZ', 'SZ1', 'SZ2', 'SZ3', 'SZ4', 'SZ5', 'T1', 'T1F', 'T2', 'T2F', 'TF2', 'TF3', 'TF4', 'TF5', 'TF6', 'TF7', 'TF8', 'TF9', 'X12', 'X13', 'XB', 'XBN', 'XC', 'XFA', 'XFB', 'XFC', 'XFZ2', 'XFZ3', 'XFZ4', 'XFZ5', 'XL1', 'XL3', 'XV', 'XVI', 'XY', 'XZ1', 'XZ2', 'XZ3', 'XZ3F', 'XZ4', 'XZ5', 'XZ7', 'XZF', 'YS', 'YSF', 'YSK', 'YXB', 'YXF']