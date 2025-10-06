import pandas as pd
import json

# Load CSV
df = pd.read_csv("./app/data/smartphones.csv")

# Create a 'features' list from boolean columns
feature_cols = ["has_5g", "has_nfc", "has_ir_blaster", "extended_memory_available"]

df["features"] = df[feature_cols].apply(
    lambda x: [col.replace("_", " ") for col, val in x.items() if val], axis=1
)

# Generate 'details' column as description
df["details"] = df.apply(
    lambda row: f"{row['brand_name']} {row['model']} runs on {row['os']} with {row['ram_capacity']} RAM, {row['battery_capacity']} mAh battery and {row['primary_camera_rear']} rear camera.", axis=1
)

# Select and rename columns
df = df.rename(columns={
    "brand_name": "brand",
    "model": "name",
    "price": "price",
    "ram_capacity": "ram",
    "battery_capacity": "battery",
    "fast_charging": "charging",
    "os": "os",
    "screen_size": "screen_size",
    "primary_camera_rear": "camera"
})

# Keep only relevant columns
df = df[["brand", "name", "price", "ram", "battery", "charging", "os", "screen_size", "camera", "features", "details"]]

# Convert to list of dicts
phones_list = df.to_dict(orient="records")

# Save as JSON
with open("./app/data/products.json", "w", encoding="utf-8") as f:
    json.dump(phones_list, f, ensure_ascii=False, indent=4)

print("Conversion complete! JSON saved to app/data/products.json")
