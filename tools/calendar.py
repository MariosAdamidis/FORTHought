"""
title: Calendar Event Creator
author: Marios Adamidis
version: 1.0.0
description: Compose calendar events and hand them off to the user's calendar app. Produces an interactive card with "Add to Google Calendar" (one-click URL, no OAuth) and ".ics" (download for Apple Calendar, Outlook, any calendar client). Default timezone Europe/Athens. Optional ORGANIZER_EMAIL UserValve (falls back to email tool's EMAIL_ADDRESS if set). MVP scope: compose only — no read/update/delete (those would need OAuth).
"""

import json
import uuid
import html as html_mod
import urllib.parse
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse


def _parse_datetime(dt_str: str):
    """
    Parse ISO 8601 datetime. Returns (datetime, is_date_only).
    Accepts:
      "2026-04-22"                    -> (dt, True)   all-day
      "2026-04-22T14:00"              -> (naive dt, False)  local, no seconds
      "2026-04-22T14:00:00"           -> (naive dt, False)
      "2026-04-22T14:00:00+03:00"     -> (aware dt, False)
      "2026-04-22T11:00:00Z"          -> (aware UTC, False)
      "2026-04-22 14:00"              -> (naive dt, False)  space tolerated
    """
    if not dt_str or not dt_str.strip():
        raise ValueError("Empty datetime string")
    s = dt_str.strip().replace(" ", "T")
    # Date-only
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return datetime.fromisoformat(s), True
    # Normalize Z for Python < 3.11 compat
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Try as-is
    try:
        return datetime.fromisoformat(s), False
    except ValueError:
        pass
    # Retry with seconds injected: 2026-04-22T14:00 or 2026-04-22T14:00+03:00
    m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})(([+-]\d{2}:?\d{2})|)$", s)
    if m:
        base = m.group(1) + ":00"
        if m.group(2):
            base += m.group(2)
        try:
            return datetime.fromisoformat(base), False
        except ValueError:
            pass
    raise ValueError(
        f"Could not parse '{dt_str}'. Use ISO 8601 like '2026-04-22T14:00:00' "
        f"or '2026-04-22T14:00:00+03:00' or '2026-04-22' (all-day)."
    )


def _ics_escape(s):
    if s is None:
        return ""
    return (
        s.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def _fold_ics_line(line):
    """RFC 5545 §3.1: lines longer than 75 octets MUST be folded."""
    if len(line) <= 75:
        return line
    out = [line[:75]]
    rest = line[75:]
    while rest:
        out.append(" " + rest[:74])
        rest = rest[74:]
    return "\r\n".join(out)


def _format_gcal_datetime(dt, is_all_day):
    if is_all_day:
        return dt.strftime("%Y%m%d")
    if dt.tzinfo:
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y%m%dT%H%M%SZ")
    # Naive: floating local time
    return dt.strftime("%Y%m%dT%H%M%S")


def _format_ics_datetime(dt, is_all_day):
    if is_all_day:
        return dt.strftime("%Y%m%d")
    if dt.tzinfo:
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y%m%dT%H%M%SZ")
    return dt.strftime("%Y%m%dT%H%M%S")


def _parse_attendees(s):
    if not s:
        return []
    return [a.strip() for a in re.split(r"[;,]", s) if a.strip()]


def _build_gcal_url(
    title,
    start_dt,
    end_dt,
    is_all_day,
    description="",
    location="",
    attendees_list=None,
):
    params = [("action", "TEMPLATE"), ("text", title)]
    start_fmt = _format_gcal_datetime(start_dt, is_all_day)
    end_fmt = _format_gcal_datetime(end_dt, is_all_day)
    params.append(("dates", f"{start_fmt}/{end_fmt}"))
    if description:
        params.append(("details", description))
    if location:
        params.append(("location", location))
    if attendees_list:
        params.append(("add", ",".join(attendees_list)))
    return (
        "https://calendar.google.com/calendar/u/0/r/eventedit?"
        + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    )


def _build_ics(
    title,
    start_dt,
    end_dt,
    is_all_day,
    description="",
    location="",
    attendees_list=None,
    organizer_email="",
):
    uid = str(uuid.uuid4()) + "@forthought"
    now_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    start_fmt = _format_ics_datetime(start_dt, is_all_day)
    end_fmt = _format_ics_datetime(end_dt, is_all_day)
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//FORTHought//Calendar Tool//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{now_utc}",
    ]
    if is_all_day:
        lines.append(f"DTSTART;VALUE=DATE:{start_fmt}")
        lines.append(f"DTEND;VALUE=DATE:{end_fmt}")
    else:
        lines.append(f"DTSTART:{start_fmt}")
        lines.append(f"DTEND:{end_fmt}")
    lines.append(f"SUMMARY:{_ics_escape(title)}")
    if description:
        lines.append(f"DESCRIPTION:{_ics_escape(description)}")
    if location:
        lines.append(f"LOCATION:{_ics_escape(location)}")
    if organizer_email:
        lines.append(f"ORGANIZER:mailto:{organizer_email}")
    if attendees_list:
        for a in attendees_list:
            lines.append(f"ATTENDEE;ROLE=REQ-PARTICIPANT;RSVP=TRUE:mailto:{a}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    folded = [_fold_ics_line(line) for line in lines]
    return "\r\n".join(folded) + "\r\n"


class Tools:
    class Valves(BaseModel):
        """Admin-only settings."""

        DEFAULT_TIMEZONE: str = Field(
            default="Europe/Athens",
            description="IANA timezone name shown as hint (not enforced — naive datetimes remain floating/local)",
        )
        DEFAULT_DURATION_MINUTES: int = Field(
            default=60,
            description="Event duration when end_time is not provided (timed events only)",
        )

    class UserValves(BaseModel):
        """Per-user: organizer email for .ics (optional)."""

        ORGANIZER_EMAIL: str = Field(
            default="",
            description="Your email — appears as ORGANIZER in .ics files. Optional; falls back to your email tool's EMAIL_ADDRESS if set there.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.citation = False

    def _get_organizer_email(self, __user__):
        """Read ORGANIZER_EMAIL from user valves; fall back to email tool's EMAIL_ADDRESS."""
        if not __user__:
            return ""
        # Path 1: OWUI middleware-populated valves
        uv = __user__.get("valves")
        if uv:
            email = (getattr(uv, "ORGANIZER_EMAIL", "") or "").strip()
            if email:
                return email
        # Path 2: DB self-load (matches email tool pattern for Gemini pipe bypass)
        user_id = __user__.get("id")
        if not user_id:
            return ""
        try:
            import sqlite3 as _sql

            _db = _sql.connect("/app/backend/data/webui.db")
            _cur = _db.cursor()
            _cur.execute("SELECT settings FROM user WHERE id=?", (user_id,))
            row = _cur.fetchone()
            _db.close()
            if row and row[0]:
                settings = json.loads(row[0])
                cal_valves = (
                    settings.get("tools", {}).get("valves", {}).get("calendar", {})
                )
                email = (cal_valves.get("ORGANIZER_EMAIL", "") or "").strip()
                if email:
                    return email
                # Reuse email tool's address as a sensible fallback
                email_valves = (
                    settings.get("tools", {}).get("valves", {}).get("email", {})
                )
                email = (email_valves.get("EMAIL_ADDRESS", "") or "").strip()
                if email:
                    return email
        except Exception:
            pass
        return ""

    async def create_event(
        self,
        title: str,
        start_time: str,
        end_time: str = "",
        description: str = "",
        location: str = "",
        attendees: str = "",
        all_day: bool = False,
        __user__: dict = None,
        __event_emitter__=None,
    ) -> HTMLResponse:
        """
        Create a calendar event draft and display it as an interactive card with two buttons:
          - "Add to Google Calendar" → opens calendar.google.com with the event pre-filled;
            user clicks Save in Google Calendar to add the event.
          - ".ics" → downloads an RFC 5545 file that opens in Apple Calendar, Outlook,
            or any calendar app (double-click the downloaded file).

        No OAuth, no credentials. User confirms the event in their calendar app — same
        safety pattern as the email tool's compose → confirm → send flow.

        Use this whenever the user asks to schedule, book, create, or add a calendar event:
        meetings, appointments, deadlines, blocks of time, reminders.

        :param title: Event title / summary (required, plain text)
        :param start_time: Start time as ISO 8601. Examples:
                           "2026-04-22T14:00:00"       (local time, interpreted in user's calendar TZ)
                           "2026-04-22T14:00:00+03:00" (explicit timezone offset)
                           "2026-04-22T11:00:00Z"      (UTC)
                           "2026-04-22"                (all-day — no time part)
                           Resolve natural language like "tomorrow 2pm" to ISO before calling.
                           Default timezone for naive datetimes is Europe/Athens.
        :param end_time:  End time as ISO 8601. If empty: +1 hour for timed events, +1 day for all-day.
        :param description: Event description / agenda / notes (plain text, newlines OK)
        :param location:    Physical address, room name, or video meeting link
        :param attendees:   Comma- or semicolon-separated attendee email addresses
        :param all_day:     Force all-day. Auto-detected when start_time has no time part.
        :return: Interactive event card with Add-to-Calendar + .ics buttons
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Composing event...", "done": False},
                }
            )

        # Parse times with graceful error card on failure
        try:
            start_dt, start_date_only = _parse_datetime(start_time)
            is_all_day = bool(all_day) or start_date_only
            if is_all_day:
                if end_time:
                    end_dt, _eo = _parse_datetime(end_time)
                else:
                    end_dt = start_dt + timedelta(days=1)
            else:
                if end_time:
                    end_dt, end_date_only = _parse_datetime(end_time)
                    if end_date_only:
                        end_dt = end_dt + timedelta(days=1)
                else:
                    end_dt = start_dt + timedelta(
                        minutes=self.valves.DEFAULT_DURATION_MINUTES
                    )
            # Sanity: end must be after start
            if end_dt <= start_dt:
                end_dt = start_dt + timedelta(hours=1)
        except ValueError as e:
            err = html_mod.escape(str(e))
            err_html = (
                '<div style="color:#f87171;padding:14px;font-family:-apple-system,sans-serif;'
                'background:#1a1b26;border:1px solid #7f1d1d;border-radius:8px;font-size:13px">'
                f"\u274c Could not parse times: {err}</div>"
            )
            return HTMLResponse(
                content=err_html, headers={"Content-Disposition": "inline"}
            )

        attendees_list = _parse_attendees(attendees)
        organizer_email = self._get_organizer_email(__user__)
        gcal_url = _build_gcal_url(
            title,
            start_dt,
            end_dt,
            is_all_day,
            description,
            location,
            attendees_list,
        )
        ics_content = _build_ics(
            title,
            start_dt,
            end_dt,
            is_all_day,
            description,
            location,
            attendees_list,
            organizer_email,
        )
        card_html = self._build_card_html(
            title,
            start_dt,
            end_dt,
            is_all_day,
            description,
            location,
            attendees_list,
            gcal_url,
            ics_content,
            organizer_email,
        )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Event draft ready", "done": True},
                }
            )

        return HTMLResponse(
            content=card_html, headers={"Content-Disposition": "inline"}
        )

    def _format_human_dt(self, dt, is_all_day):
        if is_all_day:
            return dt.strftime("%a, %b %d, %Y")
        if dt.tzinfo:
            tz_label = dt.strftime("%Z") or str(dt.tzinfo)
            return dt.strftime("%a, %b %d, %Y %H:%M ") + tz_label
        return dt.strftime("%a, %b %d, %Y %H:%M")

    def _build_card_html(
        self,
        title,
        start_dt,
        end_dt,
        is_all_day,
        description,
        location,
        attendees_list,
        gcal_url,
        ics_content,
        organizer_email,
    ):
        title_esc = html_mod.escape(title)
        location_esc = html_mod.escape(location) if location else ""
        description_html = (
            html_mod.escape(description).replace("\n", "<br>") if description else ""
        )

        # Human-readable time range
        if is_all_day:
            # DTEND is exclusive in iCal — show inclusive end
            end_display_dt = end_dt - timedelta(days=1)
            if start_dt.date() == end_display_dt.date():
                time_display = self._format_human_dt(start_dt, True) + "  (all day)"
            else:
                time_display = (
                    f"{self._format_human_dt(start_dt, True)} \u2192 "
                    f"{self._format_human_dt(end_display_dt, True)}  (all day)"
                )
        else:
            if start_dt.date() == end_dt.date():
                end_time_fmt = end_dt.strftime("%H:%M")
                if end_dt.tzinfo:
                    tz_label = end_dt.strftime("%Z") or str(end_dt.tzinfo)
                    end_time_fmt += " " + tz_label
                time_display = (
                    f"{self._format_human_dt(start_dt, False)} \u2192 {end_time_fmt}"
                )
            else:
                time_display = (
                    f"{self._format_human_dt(start_dt, False)} \u2192 "
                    f"{self._format_human_dt(end_dt, False)}"
                )
        time_esc = html_mod.escape(time_display)

        gcal_url_esc = html_mod.escape(gcal_url)
        ics_content_escaped = json.dumps(ics_content)
        ics_filename = re.sub(r"[^a-zA-Z0-9 _-]", "_", title) or "event"

        location_row = ""
        if location:
            location_row = (
                '<div class="row"><span class="lbl">Where:</span>'
                f'<span class="val">{location_esc}</span></div>'
            )

        attendees_row = ""
        if attendees_list:
            chips = "".join(
                f'<span class="chip">{html_mod.escape(a)}</span>'
                for a in attendees_list
            )
            attendees_row = (
                '<div class="row"><span class="lbl">With:</span>'
                f'<span class="chips">{chips}</span></div>'
            )

        organizer_row = ""
        if organizer_email:
            organizer_row = (
                '<div class="row"><span class="lbl">From:</span>'
                f'<span class="chip">{html_mod.escape(organizer_email)}</span></div>'
            )

        description_section = ""
        if description:
            description_section = f'<div class="body-area">{description_html}</div>'

        attendee_count = len(attendees_list)
        attendee_label = f"{attendee_count} attendee" + (
            "s" if attendee_count != 1 else ""
        )

        card_html = "".join(
            [
                '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1"><style>',
                "*{margin:0;padding:0;box-sizing:border-box}html,body{background:transparent;height:auto;overflow:visible}",
                'body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;color:#e5e7eb;line-height:1.5;padding:2px}',
                ".card{position:relative;background:#1a1b26;border:1px solid #2a2b3d;border-radius:12px;overflow:hidden}",
                '.card::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#10b981,#3b82f6,#6366f1);z-index:1}',
                ".header{display:flex;align-items:center;justify-content:space-between;padding:10px 14px 8px;border-bottom:1px solid #252636;gap:8px}",
                ".header-left{display:flex;align-items:center;gap:8px;flex:1;min-width:0}",
                ".header-title{font-weight:600;font-size:13px;color:#9ca3af}",
                ".actions{display:flex;gap:4px;flex-shrink:0}",
                ".btn{display:inline-flex;align-items:center;justify-content:center;height:30px;padding:0 12px;border:1px solid #2a2b3d;background:#1f2028;color:#9ca3af;border-radius:6px;cursor:pointer;font-size:11px;font-family:inherit;text-decoration:none;gap:4px;transition:all .15s;white-space:nowrap}.btn:hover{background:#2a2b3d;color:#e5e7eb}",
                ".btn-primary{background:#0a3d2a;color:#34d399;border-color:#065f46}.btn-primary:hover{background:#14532d;color:#6ee7b7}",
                ".fields{padding:0 14px}.row{display:flex;align-items:flex-start;gap:8px;padding:8px 0;font-size:13px}.row+.row{border-top:1px solid #252636}",
                ".lbl{font-weight:500;color:#9ca3af;min-width:56px;flex-shrink:0}",
                ".val{color:#e5e7eb;word-break:break-word}",
                ".chip{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#2a2b3d;color:#d1d5db;margin-right:4px;margin-bottom:2px}",
                ".chips{display:flex;flex-wrap:wrap;gap:2px}",
                ".subj{font-weight:600;color:#e5e7eb;word-break:break-word}",
                ".time{color:#60a5fa;font-weight:500}",
                ".body-area{padding:12px 14px;font-size:13px;line-height:1.7;color:#e5e7eb;border-top:1px solid #252636;min-height:40px;word-break:break-word}",
                ".footer{display:flex;align-items:center;justify-content:space-between;padding:8px 14px;font-size:11px;color:#6b7280;border-top:1px solid #252636;gap:8px}",
                '</style></head><body><div class="card">',
                '<div class="header"><div class="header-left">',
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="16" height="16" style="color:#9ca3af;flex-shrink:0"><path stroke-linecap="round" stroke-linejoin="round" d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 0 1 2.25-2.25h13.5A2.25 2.25 0 0 1 21 7.5v11.25m-18 0A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75m-18 0v-7.5A2.25 2.25 0 0 1 5.25 9h13.5A2.25 2.25 0 0 1 21 11.25v7.5"/></svg>',
                '<span class="header-title">Event Draft</span>',
                '</div><div class="actions">',
                '<a class="btn btn-primary" href="',
                gcal_url_esc,
                '" target="_blank" rel="noopener noreferrer" title="Opens Google Calendar with the event pre-filled. Click Save in Google Calendar to confirm.">Add to Google Calendar</a>',
                '<button class="btn" title="Download .ics \u2014 double-click the file to open in Apple Calendar, Outlook, or any calendar app (reliable fallback)" onclick="(function(){var b=new Blob([',
                ics_content_escaped,
                "],{type:'text/calendar'});var a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='",
                ics_filename,
                ".ics';a.click()})()\">.ics</button>",
                "</div></div>",
                '<div class="fields">',
                '<div class="row"><span class="lbl">Event:</span>',
                f'<span class="subj">{title_esc}</span></div>',
                '<div class="row"><span class="lbl">When:</span>',
                f'<span class="time">{time_esc}</span></div>',
                location_row,
                attendees_row,
                organizer_row,
                "</div>",
                description_section,
                '<div class="footer"><span>',
                attendee_label,
                "</span><span>",
                'Click "Add to Google Calendar" to save',
                "</span></div>",
                "</div>",
                '<script>(function(){function r(){var h=document.documentElement.scrollHeight;try{parent.postMessage({type:"iframe:height",height:h},"*")}catch(e){}}r();window.addEventListener("load",r);setTimeout(r,50);setTimeout(r,200);setTimeout(r,500);if(window.ResizeObserver)new ResizeObserver(r).observe(document.body)})()</script>',
                "</body></html>",
            ]
        )
        return card_html
