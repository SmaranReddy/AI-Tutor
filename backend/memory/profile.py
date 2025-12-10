profiles = {}

def get_profile(user_id: str):
    return profiles.get(user_id, {"mastery": 0.0})

def update_mastery(user_id: str, delta: float):
    profiles.setdefault(user_id, {"mastery": 0.0})
    profiles[user_id]["mastery"] += delta
