import re

UUID_REGEX = r'with ID ([a-f0-9\-]{36})'

def parse_ids(row):
    desc = str(row["description"])
    match = re.search(UUID_REGEX, desc, re.IGNORECASE)
    return match.group(1) if match else None
