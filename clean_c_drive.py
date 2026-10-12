import os
import shutil

user_profile = os.environ.get("USERPROFILE", r"C:\Users\ANU THOMSON")
local_app_data = os.environ.get("LOCALAPPDATA", os.path.join(user_profile, "AppData", "Local"))
roaming = os.environ.get("APPDATA", os.path.join(user_profile, "AppData", "Roaming"))

dirs_to_purge = [
    # 1. npm cache (pure package cache, 1.67 GB)
    os.path.join(local_app_data, "npm-cache"),
    # 2. Crash dumps (past crash memory dumps, 355 MB)
    os.path.join(local_app_data, "CrashDumps"),
    # 3. Pip cache
    os.path.join(local_app_data, "pip", "cache"),
    # 4. VS Code CachedData & Cache
    os.path.join(roaming, "Code", "CachedData"),
    os.path.join(roaming, "Code", "Cache"),
    # 5. General .cache
    os.path.join(user_profile, ".cache", "torch"),
    os.path.join(user_profile, ".cache", "huggingface"),
]

def purge_folder_contents(folder_path):
    deleted_bytes = 0
    if not os.path.exists(folder_path):
        return 0
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for f in files:
            fp = os.path.join(root, f)
            try:
                size = os.path.getsize(fp)
                os.remove(fp)
                deleted_bytes += size
            except (PermissionError, OSError):
                pass
        for d in dirs:
            dp = os.path.join(root, d)
            try:
                os.rmdir(dp)
            except (PermissionError, OSError):
                pass
    return deleted_bytes

print("=" * 70)
print("SAFELY CLEANING UNNECESSARY CACHE & TEMP FILES ON DRIVE C:")
print("=" * 70)

c_before = shutil.disk_usage("C:").free
total_freed = 0

# 1. Purge specific cache folders
for p in dirs_to_purge:
    if os.path.exists(p):
        freed = purge_folder_contents(p)
        mb = freed / (1024**2)
        total_freed += freed
        print(f"Purged {p:<55}: {mb:.1f} MB freed")

# 2. Clean User Temp Directory (skip in-use files)
temp_dir = os.path.join(local_app_data, "Temp")
if os.path.exists(temp_dir):
    temp_freed = 0
    for item in os.listdir(temp_dir):
        ip = os.path.join(temp_dir, item)
        try:
            if os.path.isfile(ip) or os.path.islink(ip):
                sz = os.path.getsize(ip)
                os.remove(ip)
                temp_freed += sz
            elif os.path.isdir(ip):
                temp_freed += purge_folder_contents(ip)
                try:
                    os.rmdir(ip)
                except (PermissionError, OSError):
                    pass
        except (PermissionError, OSError):
            pass
    total_freed += temp_freed
    print(f"Cleaned {temp_dir:<55}: {temp_freed / (1024**2):.1f} MB freed")

c_after = shutil.disk_usage("C:").free
print("-" * 70)
print(f"Total Bytes Cleaned        : {total_freed / (1024**3):.2f} GB")
print(f"Drive C Free Space Before  : {c_before / (1024**3):.2f} GB")
print(f"Drive C Free Space After   : {c_after / (1024**3):.2f} GB")
print(f"Net Space Reclaimed        : {(c_after - c_before) / (1024**3):.2f} GB")
print("=" * 70)

