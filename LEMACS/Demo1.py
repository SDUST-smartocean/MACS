import re
# 正则提取函数

def extract_ranges_and_multipliers(text):
    pattern_x = r'x\s*=\s*$(-?\d+\.?\d*(?:,\s*-?\d+\.?\d*)*)$'
    pattern_y = r'y\s*=\s*$(-?\d+\.?\d*(?:,\s*-?\d+\.?\d*)*)$'
    pattern_multiplier = r'probability\s*(\d+)\s*times'
    x_matches = re.findall(pattern_x, text)
    y_matches = re.findall(pattern_y, text)
    multiplier_matches = re.findall(pattern_multiplier, text)
    x_ranges = [tuple(map(float, match.split(','))) for match in x_matches if match]
    y_ranges = [tuple(map(float, match.split(','))) for match in y_matches if match]
    multipliers = [int(m) for m in multiplier_matches]
    if not (len(x_ranges) == len(y_ranges) == len(multipliers)):
        raise ValueError("Mismatch between the number of x ranges,"
                         " y ranges, and multipliers provided.")
    return [
        {"x_range": x, "y_range": y, "multiplier": m}
        for x, y, m in zip(x_ranges, y_ranges, multipliers)
    ]