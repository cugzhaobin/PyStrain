"""GPS time conversion utilities.

Ported and vectorized from the legacy PyStrain ``GPSTime.py`` (Zhao Bin, 2018).
All Julian-day / decimal-year conversions follow the algorithms tested since 2018.

"""

import numpy as np

# ---------------------------------------------------------------------------
# Core scalar helpers
# ---------------------------------------------------------------------------

def jd_to_ymdhms(jd):
    """Convert Julian day to (date, seconds, day_of_year).

    Parameters
    ----------
    jd : float
        Julian day.

    Returns
    -------
    date : np.ndarray (5,)
        [year, month, day, hour, minute].
    seconds : float
        Seconds component.
    day_of_year : int
        Day of year (1-366).
    """
    date = np.zeros(5)
    days_to_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    if jd >= 2000000.0:
        mjd = jd - 2400000.5
    else:
        mjd = jd

    mjd_day = int(mjd)
    fraction = np.mod(mjd, 1.0)

    if mjd < 0 and fraction != 0.0:
        mjd_day -= 1
        fraction += 1.0

    days_from_1600 = int(mjd_day - (-94554.0))
    years_from_1600 = int(days_from_1600 / 365.0)

    day_of_year = 0
    while day_of_year <= 0:
        century = years_from_1600 // 100
        day_of_year = (days_from_1600 - years_from_1600 * 365
                       - (years_from_1600 - 1) // 4
                       + (years_from_1600 + 99) // 100
                       - (years_from_1600 + 399) // 400 - 1)
        if years_from_1600 == 0:
            day_of_year += 1
        if day_of_year <= 0:
            years_from_1600 -= 1

    year = np.mod(years_from_1600, 100)

    leap_year = False
    if year == 0:
        if np.mod(century, 4) == 0:
            leap_year = True
    else:
        if np.mod(year, 4) == 0:
            leap_year = True

    if day_of_year < 60:
        if day_of_year <= 31:
            month = 1
            day = day_of_year
        else:
            month = 2
            day = day_of_year - 31
    else:
        if leap_year and day_of_year == 60:
            month = 2
            day = 29
        else:
            if leap_year:
                day_of_year -= 1
            month = 2
            while day_of_year > days_to_month[month - 1]:
                month += 1
            month -= 1
            day = day_of_year - days_to_month[month - 1]

    date[0] = years_from_1600 + 1600
    date[1] = month
    date[2] = day
    date[3] = fraction * 24.0
    date[4] = fraction * 1440.0 - date[3] * 60.0

    seconds = 86400.0 * fraction - date[3] * 3600.0 - date[4] * 60.0

    if seconds >= 59.0:
        dsec = 1e-6
        fracp = fraction + dsec / 86400.0
        date[3] = fracp * 24.0
        date[4] = fracp * 1440.0 - date[3] * 60.0
        seconds = 86400.0 * fracp - date[3] * 3600.0 - date[4] * 60.0 - dsec

    # Recover full day-of-year for the return value
    _, _, doy = _jd_to_ymdhms_doy(jd)

    return date, seconds, doy


def _jd_to_ymdhms_doy(jd):
    """Internal: compute only day-of-year from JD (avoids double computation)."""
    if jd >= 2000000.0:
        mjd = jd - 2400000.5
    else:
        mjd = jd

    mjd_day = int(mjd)
    fraction = np.mod(mjd, 1.0)

    if mjd < 0 and fraction != 0.0:
        mjd_day -= 1
        fraction += 1.0

    days_from_1600 = int(mjd_day - (-94554.0))
    years_from_1600 = int(days_from_1600 / 365.0)

    day_of_year = 0
    while day_of_year <= 0:
        day_of_year = (days_from_1600 - years_from_1600 * 365
                       - (years_from_1600 - 1) // 4
                       + (years_from_1600 + 99) // 100
                       - (years_from_1600 + 399) // 400 - 1)
        if years_from_1600 == 0:
            day_of_year += 1
        if day_of_year <= 0:
            years_from_1600 -= 1

    year = np.mod(years_from_1600, 100)
    century = years_from_1600 // 100

    leap_year = False
    if year == 0:
        if np.mod(century, 4) == 0:
            leap_year = True
    else:
        if np.mod(year, 4) == 0:
            leap_year = True

    date = np.array([years_from_1600 + 1600, 0, 0, 0, 0])

    days_to_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    doy = int(day_of_year)

    if day_of_year < 60:
        if day_of_year <= 31:
            month, day = 1, int(day_of_year)
        else:
            month, day = 2, int(day_of_year - 31)
    else:
        if leap_year and day_of_year == 60:
            month, day = 2, 29
        else:
            if leap_year:
                day_of_year -= 1
            month = 2
            while day_of_year > days_to_month[month - 1]:
                month += 1
            month -= 1
            day = int(day_of_year - days_to_month[month - 1])

    date[1] = month
    date[2] = day
    date[3] = fraction * 24.0
    date[4] = fraction * 1440.0 - date[3] * 60.0

    seconds = 86400.0 * fraction - date[3] * 3600.0 - date[4] * 60.0

    return date, seconds, doy


def ymdhms_to_jd(date, seconds):
    """Convert calendar date to Julian day.

    Parameters
    ----------
    date : array-like (5,)
        [year, month, day, hour, minute].
    seconds : float
        Seconds.

    Returns
    -------
    jd : float
        Julian day.
    """
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    hour = int(date[3])
    minute = int(date[4])

    days_to_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    if year < 50:
        year += 2000
    elif year < 200:
        year += 1900

    years_from_1600 = year - 1600
    leap_days = ((years_from_1600 - 1) // 4
                 - (years_from_1600 + 99) // 100
                 + (years_from_1600 + 399) // 400 + 1)
    if years_from_1600 == 0:
        leap_days -= 1

    leap_year = False
    if (np.mod(years_from_1600, 4) == 0
            and (np.mod(years_from_1600, 100) != 0
                 or np.mod(years_from_1600, 400) == 0)):
        leap_year = True

    days_from_1600 = (years_from_1600 * 365 + leap_days
                      + days_to_month[month - 1] + day)
    if month > 2 and leap_year:
        days_from_1600 += 1

    fraction = seconds / 86400.0 + minute / 1440.0 + hour / 24.0
    mjd = -94554.0 + days_from_1600 + fraction
    return mjd + 2400000.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def jd_to_decyrs(jd):
    """Convert Julian day(s) to decimal year(s).

    Parameters
    ----------
    jd : float or array_like
        Julian day(s).

    Returns
    -------
    float or np.ndarray
        Decimal year(s).
    """
    scalar = np.isscalar(jd)
    jd_arr = np.atleast_1d(np.asarray(jd, dtype=float))
    result = np.empty_like(jd_arr)

    for i in range(len(jd_arr)):
        jdi = float(jd_arr[i])

        if jdi < 2000000.0:
            jmd = jdi + 2400000.5
        else:
            jmd = jdi

        date, seconds, doy = jd_to_ymdhms(jdi)

        # JD at start of year
        date[1] = 1
        date[2] = 1
        date[3] = 0
        date[4] = 0
        jd_start = ymdhms_to_jd(date, 0.0)

        # JD at start of next year
        date[0] += 1
        jd_end = ymdhms_to_jd(date, 0.0)
        date[0] -= 1

        num_days = jd_end - jd_start
        if num_days <= 365.0 and num_days <= 366.0:
            num_days = 365.0

        result[i] = date[0] + (jmd - jd_start) / num_days

    return float(result[0]) if scalar else result


def decyrs_to_jd(decyrs):
    """Convert decimal year(s) to Julian day(s).

    Parameters
    ----------
    decyrs : float or array_like
        Decimal year(s).

    Returns
    -------
    float or np.ndarray
        Julian day(s).
    """
    scalar = np.isscalar(decyrs)
    dec_arr = np.atleast_1d(np.asarray(decyrs, dtype=float))
    result = np.empty_like(dec_arr)

    for i in range(len(dec_arr)):
        dec = float(dec_arr[i])
        date = np.zeros(5)
        date[0] = int(dec)
        date[1] = 1
        date[2] = 1

        jd_start = ymdhms_to_jd(date, 0.0)
        date[0] += 1
        jd_end = ymdhms_to_jd(date, 0.0)

        result[i] = jd_start + (dec - int(dec)) * (jd_end - jd_start)

    return float(result[0]) if scalar else result


def decyrs_to_mjd(decyrs):
    """Convert decimal year(s) to Modified Julian day(s).

    Parameters
    ----------
    decyrs : float or array_like
        Decimal year(s).

    Returns
    -------
    float or np.ndarray
        Modified Julian day(s).
    """
    jd = decyrs_to_jd(decyrs)
    return jd - 2400000.0


def ymd_to_decyrs(year, month, day):
    """Convert YYYY-MM-DD to decimal year.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    float
        Decimal year.
    """
    jdi = ymdhms_to_jd([year, month, day, 0, 0], 0.0)

    if jdi < 2000000.0:
        jdm = jdi + 2400000.5
    else:
        jdm = jdi

    jd_start = ymdhms_to_jd([year, 1, 1, 0, 0], 0.0)
    jd_end = ymdhms_to_jd([year + 1, 1, 1, 0, 0], 0.0)

    num_days = jd_end - jd_start
    if num_days <= 365.0 and num_days <= 366.0:
        num_days = 365.0

    return year + (jdm - jd_start) / num_days


def daily_epoch_grid(sepoch, eepoch):
    """Generate a daily decimal-year epoch grid between two epochs.

    Parameters
    ----------
    sepoch : float
        Start decimal year.
    eepoch : float
        End decimal year.

    Returns
    -------
    np.ndarray
        1-D array of daily decimal-year epochs.
    """
    smjd = decyrs_to_mjd(sepoch)
    emjd = decyrs_to_mjd(eepoch)
    mjd = np.arange(smjd, emjd)
    decyrs = np.array([jd_to_decyrs(m + 2400000.0) for m in mjd])
    return decyrs
