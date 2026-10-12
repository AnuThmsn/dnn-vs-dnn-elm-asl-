import os
import shutil

user_profile = os.environ.get("USERPROFILE", r"C:\Users\ANU THOMSON")
local_app_data = os.environ.get("LOCALAPPDATA", os.path.join(user_profile, "AppData", "Local"))

targets = {
    "User Temp Directory": os.path.join(local_app_data, "Temp"),
    "Pip Cache": os.path.join(local_app_data, "pip", "cache"),
    "PyTorch Cache": os.path.join(user_profile, ".cache", "torch"),
    "HuggingFace Cache": os.path.join(user_profile, ".cache", "huggingface"),
    "General User .cache": os.path.join(user_profile, ".cache"),
    "Windows CrashDumps": os.path.join(local_app_data, "CrashDumps"),
    "NPM Cache": os.path.join(local_app_data, "npm-cache"),
    "Yarn Cache": os.path.join(local_app_data, "Yarn"),
    "VS Code CachedData": os.path.join(user_profile, "AppData", "Roaming", "Code", "CachedData"),
    "VS Code Cache": os.path.join(user_profile, "AppData", "Roaming", "Code", "Cache"),
    "VS Code Service Worker Cache": os.path.join(user_profile, "AppData", "Roaming", "Code", "Service Worker", "CacheStorage"),
    "Chrome Cache": os.path.join(local_app_data, "Google", "Chrome", "User Data", "Default", "Cache"),
    "Edge Cache": os.path.join(local_app_data, "Microsoft", "Edge", "User Data", "Default", "Cache"),
}

def get_dir_size(path):
    if not os.path.exists(path):
        return 0
    total = 0
    try:
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return total

print("=" * 70)
print("SCANNING CLEANABLE CACHE & TEMP DIRECTORIES ON DRIVE C:")
print("=" * 70)

total_cleanable = 0
results = []
for name, path in targets.items():
    if os.path.exists(path):
        size_bytes = get_dir_size(path)
        size_mb = size_bytes / (1024 ** 2)
        size_gb = size_bytes / (1024 ** 3)
        results.append((name, path, size_mb, size_gb))
        total_cleanable += size_bytes

results.sort(key=lambda x: x[2], reverse=True)
for name, path, mb, gb in results:
    if mb > 1.0:
        if gb >= 1.0:
            print(f"{name:<30}: {gb:.2f} GB  ({path})")
        else:
            print(f"{name:<30}: {mb:.1f} MB  ({path})")

print("-" * 70)
print(f"Total cleanable identified: {total_cleanable / (1024**3):.2f} GB")

# Also check Downloads folder size
downloads = os.path.join(user_profile, "Downloads")
if os.path.exists(downloads):
    dl_size = get_dir_size(downloads)
    print(f"User Downloads Folder (Review only) : {dl_size / (1024**3):.2f} GB ({downloads})")

c_stat = shutil.disk_usage("C:")
print(f"Current Drive C Free Space: {c_stat.free / (1024**3):.2f} GB / {c_stat.total / (1024**3):.2f} GB")

