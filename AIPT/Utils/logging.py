from datetime import datetime
from pytz import timezone

def today(tz=None):
    return datetime.now(tz=timezone(tz)).strftime("%Y-%m-%d")

def current_time(tz=None):
    return datetime.now(tz=timezone(tz)).strftime("%H.%M.%S")
