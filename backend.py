import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

#Obtain credentials
cred = credentials.Certificate("/path/to/credentials.json")

#Set up Firebase Database URL
firebase_admin.initialize_app(cred, {
    'databaseURL': 'INSERT DATABASE URL HERE',
})

swipe_ref = db.reference("swipe_data/user_1")

#Fetch data from Firebase
swipe_events = swipe_ref.get()

#Convert data to a structured format
swipe_list = []

#If database branch found
if swipe_events:
    for event_id, event_data in swipe_events.items():
        swipe_direction = event_data.get("swipe_direction", "")
        swipe_velocity = event_data.get("swipe_velocity", 0)
        duration = event_data.get("duration", 0)
        timestamp = event_data.get("timestamp", "")

        # Extract touch start/end points safely
        touch_start = event_data.get("touch_start", {})
        touch_end = event_data.get("touch_end", {})

        # Ensure values are dictionaries
        touch_start_x = touch_start.get("x", 0)
        touch_start_y = touch_start.get("y", 0)
        touch_end_x = touch_end.get("x", 0)
        touch_end_y = touch_end.get("y", 0)

        # Append structured data
        swipe_list.append({
            "event_id": event_id,
            "swipe_direction": swipe_direction,
            "swipe_velocity": swipe_velocity,
            "touch_start_x": touch_start_x,
            "touch_start_y": touch_start_y,
            "touch_end_x": touch_end_x,
            "touch_end_y": touch_end_y,
            "duration": duration,
            "timestamp": timestamp
        })

df = pd.DataFrame(swipe_list)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("\nSwipe Data:")
print(df)

df.to_csv("swipe_data.csv", index=False)
print("\nCleaned data saved to swipe_data_cleaned.csv")

### Normalize & Encode Data ###
df.fillna({
    "swipe_direction": "unknown",
    "swipe_velocity": 0,
    "touch_start_x": 0,
    "touch_start_y": 0,
    "touch_end_x": 0,
    "touch_end_y": 0,
    "duration": 0
}, inplace=True)

label_encoder = LabelEncoder()
df["swipe_direction"] = label_encoder.fit_transform(df["swipe_direction"])

# Choose normalization method
NORMALIZATION_TYPE = "z-score"  # Options:"minmax", "z-score", "log"

# Apply normalization
if NORMALIZATION_TYPE == "minmax":
    scaler = MinMaxScaler()
    df[["swipe_velocity", "touch_start_x", "touch_start_y", "touch_end_x", "touch_end_y", "duration"]] = scaler.fit_transform(
        df[["swipe_velocity", "touch_start_x", "touch_start_y", "touch_end_x", "touch_end_y", "duration"]]
    )
elif NORMALIZATION_TYPE == "z-score":
    scaler = StandardScaler()
    df[["swipe_velocity", "touch_start_x", "touch_start_y", "touch_end_x", "touch_end_y", "duration"]] = scaler.fit_transform(
        df[["swipe_velocity", "touch_start_x", "touch_start_y", "touch_end_x", "touch_end_y", "duration"]]
    )
elif NORMALIZATION_TYPE == "log":
    df[["swipe_velocity", "touch_start_x", "touch_start_y", "touch_end_x", "touch_end_y", "duration"]] = np.log1p(
        df[["swipe_velocity", "touch_start_x", "touch_start_y", "touch_end_x", "touch_end_y", "duration"]]
    )

df["swipe_velocity"] = np.clip(df["swipe_velocity"], -1000, 1000)
df["duration"] = np.clip(df["duration"], 0, 5)

print("\nPreprocessed Swipe Data:")
print(df)

# Save preprocessed data
df.to_csv("swipe_data_preprocessed.csv", index=False)
print("\nPreprocessed data saved to swipe_data_preprocessed.csv")
