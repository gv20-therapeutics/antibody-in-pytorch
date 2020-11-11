from datetime import datetime
from pytz import timezone

def today(tz='EST'):
    return datetime.now(tz=timezone(tz)).strftime("%Y-%m-%d")

def current_time(tz='EST'):
    return datetime.now(tz=timezone(tz)).strftime("%H.%M.%S")
