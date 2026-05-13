"""
title: Email Composer & Manager
author: yunfeizhu (original MCP server), Marios Adamidis (OWUI tool adaptation)
author_url: https://github.com/yunfeizhu/mcp-mail-server
version: 5.3.0
description: Full email client with per-user credentials. Each user sets their own email/password in chat settings. Supports compose, send, read inbox, search. List/search cap = 1000. Mail App button uses target=_blank + tooltip guides user to chrome://settings/handlers if Chrome intercepts mailto.
"""

import json
import uuid
import html as html_mod
import urllib.parse
import re
import smtplib
import imaplib
import email as email_lib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from email.utils import parsedate_to_datetime
import ssl
from typing import Optional
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse


def _decode_header_value(raw):
    if not raw:
        return ""
    parts = decode_header(raw)
    decoded = []
    for data, charset in parts:
        if isinstance(data, bytes):
            decoded.append(data.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(data)
    return " ".join(decoded)


def _get_text_from_message(msg):
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/html" and "attachment" not in cd:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    html_text = payload.decode(charset, errors="replace")
                    clean = re.sub(
                        r"<style[^>]*>.*?</style>", "", html_text, flags=re.DOTALL
                    )
                    clean = re.sub(r"<[^>]+>", " ", clean)
                    clean = re.sub(r"\s+", " ", clean).strip()
                    return clean
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return "(no body)"


class Tools:
    class Valves(BaseModel):
        """Admin-only settings: server configuration shared by all users."""

        IMAP_HOST: str = Field(
            default="imap.example.com", description="IMAP server hostname"
        )
        IMAP_PORT: int = Field(default=993, description="IMAP server port")
        SMTP_HOST: str = Field(
            default="smtp.example.com", description="SMTP server hostname"
        )
        SMTP_PORT: int = Field(
            default=465, description="SMTP server port (465=SSL, 587=STARTTLS)"
        )
        SMTP_USE_SSL: bool = Field(default=True, description="Use SSL for SMTP")
        SEND_ENABLED: bool = Field(
            default=True, description="Enable sending for all users"
        )

    class UserValves(BaseModel):
        """Per-user settings: each user sets their own email and password."""

        EMAIL_ADDRESS: str = Field(
            default="",
            description="Your email address (e.g. user@example.com)",
        )
        EMAIL_PASSWORD: str = Field(default="", description="Your email password")

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.citation = False

    def _get_user_creds(self, __user__: dict = None):
        """Get email/password from user valves. Self-loads from DB if pipe doesn't provide them."""
        if not __user__:
            return None, None

        # Path 1: OWUI native middleware already populated valves
        uv = __user__.get("valves")
        if uv:
            email_addr = getattr(uv, "EMAIL_ADDRESS", "") or ""
            email_pass = getattr(uv, "EMAIL_PASSWORD", "") or ""
            if email_addr and email_pass:
                return email_addr.strip(), email_pass.strip()

        # Path 2: Gemini pipe bypasses middleware — load valves from DB directly
        user_id = __user__.get("id")
        if user_id:
            try:
                import sqlite3 as _sql

                _db = _sql.connect("/app/backend/data/webui.db")
                _cur = _db.cursor()
                _cur.execute("SELECT settings FROM user WHERE id=?", (user_id,))
                row = _cur.fetchone()
                _db.close()
                if row and row[0]:
                    import json as _json

                    settings = _json.loads(row[0])
                    tool_valves = (
                        settings.get("tools", {}).get("valves", {}).get("email", {})
                    )
                    email_addr = tool_valves.get("EMAIL_ADDRESS", "")
                    email_pass = tool_valves.get("EMAIL_PASSWORD", "")
                    if email_addr and email_pass:
                        return email_addr.strip(), email_pass.strip()
            except Exception:
                pass

        return None, None

    def _connect_imap(self, email_addr: str, email_pass: str):
        context = ssl.create_default_context()
        imap = imaplib.IMAP4_SSL(
            self.valves.IMAP_HOST, self.valves.IMAP_PORT, ssl_context=context
        )
        imap.login(email_addr, email_pass)
        return imap

    async def list_inbox(
        self,
        count: int = 10,
        unread_only: bool = False,
        __user__: dict = None,
        __event_emitter__=None,
    ) -> str:
        """
        List recent emails from the inbox. Shows sender, subject, date, and read status.
        Use this when the user asks to check their email, see recent messages, or check their inbox.

        :param count: Number of emails to show (default 10, max 1000)
        :param unread_only: If true, show only unread emails
        :return: Formatted list of recent emails
        """
        email_addr, email_pass = self._get_user_creds(__user__)
        if not email_addr or not email_pass:
            return "\u274c Email not configured. Go to **Chat Settings (gear icon) \u2192 Tools \u2192 Email Composer & Manager \u2192 Set your EMAIL_ADDRESS and EMAIL_PASSWORD**."
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Checking inbox...", "done": False},
                }
            )
        try:
            imap = self._connect_imap(email_addr, email_pass)
            imap.select("INBOX", readonly=True)
            count = min(count, 1000)
            if unread_only:
                status, data = imap.search(None, "UNSEEN")
            else:
                status, data = imap.search(None, "ALL")
            if status != "OK" or not data[0]:
                imap.logout()
                return "No emails found."
            msg_ids = data[0].split()
            recent_ids = msg_ids[-count:]
            recent_ids.reverse()
            results = []
            for msg_id in recent_ids:
                status, msg_data = imap.fetch(
                    msg_id, "(FLAGS BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
                )
                if status != "OK":
                    continue
                flags_raw = ""
                header_raw = b""
                for part in msg_data:
                    if isinstance(part, tuple):
                        if b"FLAGS" in part[0]:
                            flags_raw = part[0].decode("utf-8", errors="replace")
                        header_raw = part[1]
                    elif isinstance(part, bytes) and b"FLAGS" in part:
                        flags_raw = part.decode("utf-8", errors="replace")
                msg = email_lib.message_from_bytes(header_raw)
                from_addr = _decode_header_value(msg.get("From", ""))
                subject = _decode_header_value(msg.get("Subject", "(no subject)"))
                date_str = msg.get("Date", "")
                is_read = "\\Seen" in flags_raw
                status_icon = "  " if is_read else "\U0001f535"
                if len(from_addr) > 50:
                    from_addr = from_addr[:47] + "..."
                if len(subject) > 70:
                    subject = subject[:67] + "..."
                uid = msg_id.decode()
                results.append(
                    f"{status_icon} **[{uid}]** {subject}\n   From: {from_addr} | {date_str}"
                )
            imap.logout()
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Found {len(results)} emails",
                            "done": True,
                        },
                    }
                )
            header = f"\U0001f4ec **Inbox ({email_addr})** \u2014 {len(results)} most recent {'unread ' if unread_only else ''}emails\n\n---\n\n"
            return (
                header
                + "\n\n".join(results)
                + "\n\n---\n*Use `read_email` with the [ID] number to read the full email.*"
            )
        except imaplib.IMAP4.error as e:
            if "AUTHENTICATIONFAILED" in str(e):
                return "\u274c Login failed. Check your EMAIL_ADDRESS and EMAIL_PASSWORD in Chat Settings \u2192 Tools."
            return f"\u274c IMAP error: {e}"
        except Exception as e:
            return f"\u274c Error checking inbox: {e}"

    async def read_email(
        self, email_id: str, __user__: dict = None, __event_emitter__=None
    ) -> str:
        """
        Read the full content of a specific email by its ID number.
        Get the ID from list_inbox or search_emails results.

        :param email_id: The email ID number from inbox listing (e.g. "8917")
        :return: Full email content with headers and body
        """
        email_addr, email_pass = self._get_user_creds(__user__)
        if not email_addr or not email_pass:
            return "\u274c Email not configured. Set EMAIL_ADDRESS and EMAIL_PASSWORD in Chat Settings \u2192 Tools."
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Reading email...", "done": False},
                }
            )
        try:
            imap = self._connect_imap(email_addr, email_pass)
            imap.select("INBOX", readonly=True)
            status, msg_data = imap.fetch(email_id.encode(), "(BODY.PEEK[])")
            if status != "OK":
                imap.logout()
                return f"Email {email_id} not found."
            raw_email = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw_email)
            from_addr = _decode_header_value(msg.get("From", ""))
            to_addr = _decode_header_value(msg.get("To", ""))
            cc_addr = _decode_header_value(msg.get("Cc", ""))
            subject = _decode_header_value(msg.get("Subject", "(no subject)"))
            date_str = msg.get("Date", "")
            body = _get_text_from_message(msg)
            attachments = []
            if msg.is_multipart():
                for part in msg.walk():
                    cd = str(part.get("Content-Disposition", ""))
                    if "attachment" in cd:
                        fname = _decode_header_value(part.get_filename() or "unnamed")
                        attachments.append(fname)
            imap.logout()
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Email loaded", "done": True},
                    }
                )
            result = f"## \u2709\ufe0f Email [{email_id}]\n\n"
            result += f"**From:** {from_addr}\n"
            result += f"**To:** {to_addr}\n"
            if cc_addr:
                result += f"**Cc:** {cc_addr}\n"
            result += f"**Subject:** {subject}\n"
            result += f"**Date:** {date_str}\n"
            if attachments:
                result += f"**Attachments:** {', '.join(attachments)}\n"
            result += "\n---\n\n"
            if len(body) > 5000:
                result += (
                    body[:5000]
                    + "\n\n*... (truncated, full email is "
                    + str(len(body))
                    + " chars)*"
                )
            else:
                result += body
            return result
        except Exception as e:
            return f"\u274c Error reading email: {e}"

    async def search_emails(
        self,
        query: str,
        search_in: str = "all",
        count: int = 10,
        __user__: dict = None,
        __event_emitter__=None,
    ) -> str:
        """
        Search emails by keyword. Searches subject, sender, and body.
        Use this when the user asks to find specific emails.

        :param query: Search keyword or phrase
        :param search_in: Where to search: "subject", "from", "body", or "all" (default)
        :param count: Max results to return (default 10, max 1000)
        :return: List of matching emails
        """
        email_addr, email_pass = self._get_user_creds(__user__)
        if not email_addr or not email_pass:
            return "\u274c Email not configured. Set EMAIL_ADDRESS and EMAIL_PASSWORD in Chat Settings \u2192 Tools."
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Searching for '{query}'...",
                        "done": False,
                    },
                }
            )
        try:
            imap = self._connect_imap(email_addr, email_pass)
            imap.select("INBOX", readonly=True)
            if search_in == "subject":
                criteria = f'(SUBJECT "{query}")'
            elif search_in == "from":
                criteria = f'(FROM "{query}")'
            elif search_in == "body":
                criteria = f'(BODY "{query}")'
            else:
                criteria = f'(OR OR SUBJECT "{query}" FROM "{query}" BODY "{query}")'
            status, data = imap.search(None, criteria)
            if status != "OK" or not data[0]:
                imap.logout()
                return f"No emails found matching '{query}'."
            msg_ids = data[0].split()
            recent_ids = msg_ids[-min(count, 1000) :]
            recent_ids.reverse()
            results = []
            for msg_id in recent_ids:
                status, msg_data = imap.fetch(
                    msg_id, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
                )
                if status != "OK":
                    continue
                header_raw = b""
                for part in msg_data:
                    if isinstance(part, tuple):
                        header_raw = part[1]
                msg = email_lib.message_from_bytes(header_raw)
                from_addr = _decode_header_value(msg.get("From", ""))
                subject = _decode_header_value(msg.get("Subject", "(no subject)"))
                date_str = msg.get("Date", "")
                if len(from_addr) > 50:
                    from_addr = from_addr[:47] + "..."
                if len(subject) > 70:
                    subject = subject[:67] + "..."
                uid = msg_id.decode()
                results.append(
                    f"**[{uid}]** {subject}\n   From: {from_addr} | {date_str}"
                )
            imap.logout()
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Found {len(results)} results",
                            "done": True,
                        },
                    }
                )
            header = f'\U0001f50d **Search results for** "{query}" \u2014 {len(results)} emails found\n\n---\n\n'
            return (
                header
                + "\n\n".join(results)
                + "\n\n---\n*Use `read_email` with the [ID] number to read the full email.*"
            )
        except Exception as e:
            return f"\u274c Search error: {e}"

    async def compose_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
        priority: str = "normal",
        __user__: dict = None,
        __event_emitter__=None,
    ) -> HTMLResponse:
        """
        Composes and displays an email as an interactive card embedded in the chat.
        Use this tool whenever the user asks to write, draft, or compose an email.
        After displaying the card, ask the user if they want to send it.

        :param to: Recipient email address(es), separated by semicolons for multiple
        :param subject: Email subject line
        :param body: Plain text email body
        :param cc: CC recipient(s), separated by semicolons (optional)
        :param bcc: BCC recipient(s), separated by semicolons (optional)
        :param priority: Email priority: high, normal, or low (optional)
        :return: Interactive email card
        """
        email_addr, _ = self._get_user_creds(__user__)
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Composing email...", "done": False},
                }
            )
        card_html = self._build_card_html(
            to, subject, body, cc, bcc, priority, email_addr
        )
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Email draft ready", "done": True},
                }
            )
        return HTMLResponse(
            content=card_html, headers={"Content-Disposition": "inline"}
        )

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
        __user__: dict = None,
        __event_emitter__=None,
    ) -> str:
        """
        Sends an email directly via SMTP. Only call this AFTER the user has reviewed the draft and confirmed.
        Do NOT call this without user confirmation.

        :param to: Recipient email address(es), separated by semicolons
        :param subject: Email subject line
        :param body: Plain text email body
        :param cc: CC recipient(s), separated by semicolons (optional)
        :param bcc: BCC recipient(s), separated by semicolons (optional)
        :return: Confirmation message
        """
        if not self.valves.SEND_ENABLED:
            return "Sending disabled by admin."
        email_addr, email_pass = self._get_user_creds(__user__)
        if not email_addr or not email_pass:
            return "\u274c Email not configured. Set EMAIL_ADDRESS and EMAIL_PASSWORD in Chat Settings \u2192 Tools."
        if __event_emitter__:
            await __event_emitter__(
                {"type": "status", "data": {"description": "Sending...", "done": False}}
            )
        try:
            to_addrs = [a.strip() for a in to.replace(",", ";").split(";") if a.strip()]
            cc_addrs = (
                [a.strip() for a in cc.replace(",", ";").split(";") if a.strip()]
                if cc
                else []
            )
            bcc_addrs = (
                [a.strip() for a in bcc.replace(",", ";").split(";") if a.strip()]
                if bcc
                else []
            )
            all_recipients = to_addrs + cc_addrs + bcc_addrs
            if not all_recipients:
                return "Error: No recipients."
            msg = MIMEMultipart("alternative")
            msg["From"] = email_addr
            msg["To"] = ", ".join(to_addrs)
            if cc_addrs:
                msg["Cc"] = ", ".join(cc_addrs)
            msg["Subject"] = subject
            plain_body = body.replace("\\n", "\n")
            msg.attach(MIMEText(plain_body, "plain", "utf-8"))
            html_body = html_mod.escape(plain_body).replace("\n", "<br>")
            html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
            html_content = f'<html><body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:14px;line-height:1.6;color:#222">{html_body}</body></html>'
            msg.attach(MIMEText(html_content, "html", "utf-8"))
            context = ssl.create_default_context()
            if self.valves.SMTP_USE_SSL:
                with smtplib.SMTP_SSL(
                    self.valves.SMTP_HOST,
                    self.valves.SMTP_PORT,
                    context=context,
                    timeout=30,
                ) as server:
                    server.login(email_addr, email_pass)
                    server.sendmail(email_addr, all_recipients, msg.as_string())
            else:
                with smtplib.SMTP(
                    self.valves.SMTP_HOST, self.valves.SMTP_PORT, timeout=30
                ) as server:
                    server.starttls(context=context)
                    server.login(email_addr, email_pass)
                    server.sendmail(email_addr, all_recipients, msg.as_string())
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Email sent \u2713", "done": True},
                    }
                )
            return f"\u2705 Email sent from {email_addr} to {', '.join(to_addrs)}. Subject: {subject}"
        except smtplib.SMTPAuthenticationError:
            return "\u274c SMTP auth failed. Check your EMAIL_PASSWORD in Chat Settings \u2192 Tools."
        except Exception as e:
            return f"\u274c Failed to send: {e}"

    def _build_card_html(self, to, subject, body, cc, bcc, priority, sender_email=""):
        to_esc = html_mod.escape(to)
        subject_esc = html_mod.escape(subject)
        cc_esc = html_mod.escape(cc)
        bcc_esc = html_mod.escape(bcc)
        body_html = html_mod.escape(body).replace("\\n", "<br>").replace("\n", "<br>")
        body_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body_html)
        body_html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", body_html)
        body_html = re.sub(r"^- (.+)", r"<li>\1</li>", body_html, flags=re.MULTILINE)
        params = {}
        if subject:
            params["subject"] = subject
        if body:
            params["body"] = body.replace("\\n", "\n")
        if cc:
            params["cc"] = cc
        if bcc:
            params["bcc"] = bcc
        mailto = "mailto:" + to
        if params:
            mailto += "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        mailto_esc = html_mod.escape(mailto)
        eml_lines = ["To: " + to]
        if sender_email:
            eml_lines.insert(0, "From: " + sender_email)
        if cc:
            eml_lines.append("Cc: " + cc)
        if bcc:
            eml_lines.append("Bcc: " + bcc)
        eml_lines += [
            "Subject: " + subject,
            "MIME-Version: 1.0",
            "Content-Type: text/plain; charset=utf-8",
            "",
            body.replace("\\n", "\n"),
        ]
        eml_content_escaped = json.dumps("\r\n".join(eml_lines))
        eml_filename = re.sub(r"[^a-zA-Z0-9 _-]", "_", subject or "email")
        priority_html = ""
        if priority == "high":
            priority_html = '<span style="font-size:11px;font-weight:600;padding:2px 8px;border-radius:10px;background:#fef2f2;color:#dc2626;border:1px solid #fecaca;margin-left:8px">High Priority</span>'
        elif priority == "low":
            priority_html = '<span style="font-size:11px;font-weight:600;padding:2px 8px;border-radius:10px;background:#f9fafb;color:#9ca3af;border:1px solid #e5e7eb;margin-left:8px">Low Priority</span>'
        cc_row = ""
        if cc:
            cc_row = (
                '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;font-size:13px;border-top:1px solid #252636"><span style="font-weight:500;color:#9ca3af;min-width:48px">Cc:</span><span style="display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#2a2b3d;color:#d1d5db">'
                + cc_esc
                + "</span></div>"
            )
        bcc_row = ""
        if bcc:
            bcc_row = (
                '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;font-size:13px;border-top:1px solid #252636"><span style="font-weight:500;color:#9ca3af;min-width:48px">Bcc:</span><span style="display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#2a2b3d;color:#d1d5db">'
                + bcc_esc
                + "</span></div>"
            )
        words = len(body.split()) if body.strip() else 0
        from_row = ""
        if sender_email:
            from_esc = html_mod.escape(sender_email)
            from_row = (
                '<div class="row"><span class="lbl">From:</span><span class="chip">'
                + from_esc
                + "</span></div>"
            )
        send_badge = (
            '<span style="font-size:10px;padding:2px 6px;border-radius:8px;background:#065f4620;color:#34d399;border:1px solid #065f4640;margin-left:auto">SMTP Ready</span>'
            if self.valves.SEND_ENABLED and sender_email
            else '<span style="font-size:10px;padding:2px 6px;border-radius:8px;background:#7f1d1d20;color:#f87171;border:1px solid #7f1d1d40;margin-left:auto">Not Configured</span>'
        )
        card_html = "".join(
            [
                '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><style>',
                "*{margin:0;padding:0;box-sizing:border-box}html,body{background:transparent;height:auto;overflow:visible}",
                'body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;color:#e5e7eb;line-height:1.5;padding:2px}',
                ".card{position:relative;background:#1a1b26;border:1px solid #2a2b3d;border-radius:12px;overflow:hidden}",
                '.card::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#6366f1,#8b5cf6,#a78bfa);z-index:1}',
                ".header{display:flex;align-items:center;justify-content:space-between;padding:10px 14px 8px;border-bottom:1px solid #252636}",
                ".header-left{display:flex;align-items:center;gap:8px;flex:1}.header-title{font-weight:600;font-size:13px;color:#9ca3af}",
                ".actions{display:flex;gap:4px}",
                ".btn{display:inline-flex;align-items:center;justify-content:center;height:30px;padding:0 10px;border:1px solid #2a2b3d;background:#1f2028;color:#9ca3af;border-radius:6px;cursor:pointer;font-size:11px;font-family:inherit;text-decoration:none;gap:4px;transition:all .15s}.btn:hover{background:#2a2b3d;color:#e5e7eb}",
                ".fields{padding:0 14px}.row{display:flex;align-items:center;gap:8px;padding:8px 0;font-size:13px}.row+.row{border-top:1px solid #252636}",
                ".lbl{font-weight:500;color:#9ca3af;min-width:48px}.chip{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#2a2b3d;color:#d1d5db}",
                ".subj{font-weight:600;color:#e5e7eb}",
                ".body-area{padding:12px 14px;font-size:13px;line-height:1.7;color:#e5e7eb;border-top:1px solid #252636;min-height:60px}.body-area li{margin-left:16px;list-style:disc}",
                ".footer{display:flex;align-items:center;justify-content:space-between;padding:8px 14px;font-size:11px;color:#6b7280;border-top:1px solid #252636}",
                '</style></head><body><div class="card">',
                '<div class="header"><div class="header-left">',
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="16" height="16" style="color:#9ca3af"><path stroke-linecap="round" stroke-linejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 0 1-2.25 2.25h-15a2.25 2.25 0 0 1-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0 0 19.5 4.5h-15a2.25 2.25 0 0 0-2.25 2.25m19.5 0v.243a2.25 2.25 0 0 1-1.07 1.916l-7.5 4.615a2.25 2.25 0 0 1-2.36 0L3.32 8.91a2.25 2.25 0 0 1-1.07-1.916V6.75"/></svg>',
                '<span class="header-title">Email Draft</span>',
                priority_html,
                send_badge,
                '</div><div class="actions">',
                '<a class="btn" href="',
                mailto_esc,
                '" target="_blank" rel="noopener noreferrer" title="Open in default mail app. If Chrome opens Gmail or a blank tab instead of Mail.app: go to chrome://settings/handlers and remove (or edit to Ask) the mailto entry. Alternative: use the .eml button.">Mail App</a>',
                '<button class="btn" title="Download .eml \u2014 double-click the downloaded file to open in Mail.app (reliable fallback)" onclick="(function(){var b=new Blob([',
                eml_content_escaped,
                "],{type:'message/rfc822'});var a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='",
                eml_filename,
                ".eml';a.click()})()\">",
                ".eml</button>",
                "</div></div>",
                '<div class="fields">',
                from_row,
                '<div class="row"><span class="lbl">To:</span><span class="chip">',
                to_esc,
                "</span></div>",
                cc_row,
                bcc_row,
                '<div class="row" style="border-top:1px solid #252636"><span class="lbl">Subject:</span><span class="subj">',
                subject_esc,
                "</span></div>",
                "</div>",
                '<div class="body-area">',
                body_html,
                "</div>",
                '<div class="footer"><span>',
                str(words),
                ' words</span><span>Say "send it" to send via SMTP</span></div>',
                "</div>",
                '<script>(function(){function r(){var h=document.documentElement.scrollHeight;try{parent.postMessage({type:"iframe:height",height:h},"*")}catch(e){}}r();window.addEventListener("load",r);setTimeout(r,50);setTimeout(r,200);setTimeout(r,500);if(window.ResizeObserver)new ResizeObserver(r).observe(document.body)})()</script>',
                "</body></html>",
            ]
        )
        return card_html
