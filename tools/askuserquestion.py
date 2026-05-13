"""
title: Claude-like Ask User Question
description: Interactive MCQ tool — the LLM can ask the user a question with clickable option buttons, multi-select, drag-to-rank prioritization, free-text input, and optional per-option descriptions. Renders a polished animated overlay via OWUI's execute event.
author: Marios Adamidis
version: 1.0.0
required_open_webui_version: 0.8.0
"""

import json
import logging
from pydantic import BaseModel, Field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ask_user_question")


class Tools:
    class Valves(BaseModel):
        accent_color: str = Field(
            default="#E8713A",
            description="Accent color for selected/highlighted options (hex). Default: Claude orange.",
        )
        allow_multi_select: bool = Field(
            default=False,
            description="If true, the default mode allows multiple selections. The LLM can override per-call.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False

    def _build_mcq_js(
        self,
        question: str,
        options: list[str],
        descriptions: list[str] | None = None,
        multi_select: bool = False,
        allow_custom: bool = True,
        required: bool = False,
    ) -> str:
        """
        Build the complete JavaScript for the MCQ overlay widget.

        The JS is a self-contained IIFE that returns a Promise which resolves to a
        JSON string of the form:
          {"type":"select","indices":[0],"values":["Option A"]}
          {"type":"select","indices":[1,3],"values":["Option B","Option D"]}
          {"type":"custom","value":"user typed this"}
          {"type":"skip"}

        All user-supplied strings are set via textContent (never innerHTML) to prevent XSS.
        """
        accent = self.valves.accent_color
        q_safe = json.dumps(question)
        opts_safe = json.dumps(options)
        descs_safe = json.dumps(descriptions or [])
        multi_js = "true" if multi_select else "false"
        custom_js = "true" if allow_custom else "false"
        required_js = "true" if required else "false"
        accent_safe = json.dumps(accent)

        return f"""
return (function() {{
  return new Promise((resolve) => {{

    // ── Data injected by Python ────────────────────────────────────────────
    const question    = {q_safe};
    const options     = {opts_safe};
    const descriptions = {descs_safe};   // optional subtitle per option
    const multiSelect = {multi_js};
    const allowCustom = {custom_js};
    const required    = {required_js};   // hide Skip when true
    const accent      = {accent_safe};
    const OVERLAY_ID  = '__owui_ask_user_q__';

    // ── Remove any existing overlay to prevent stacking ───────────────────
    const existing = document.getElementById(OVERLAY_ID);
    if (existing) existing.remove();

    // ── State ─────────────────────────────────────────────────────────────
    const selectedIndices = new Set();
    let customFocused = false;   // suppress kbd shortcuts while typing

    // ── Exit animation + resolve ──────────────────────────────────────────
    function finish(payload) {{
      panel.style.transform   = 'scale(0.95)';
      panel.style.opacity     = '0';
      overlay.style.opacity   = '0';
      setTimeout(() => {{
        overlay.remove();
        document.removeEventListener('keydown', onKey);
        resolve(JSON.stringify(payload));
      }}, 180);
    }}

    // ── Overlay backdrop ──────────────────────────────────────────────────
    const overlay = document.createElement('div');
    overlay.id = OVERLAY_ID;
    Object.assign(overlay.style, {{
      position: 'fixed',
      inset: '0',
      zIndex: '999999',
      background: 'rgba(0,0,0,0.62)',
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif",
      opacity: '0',
      transition: 'opacity 0.18s ease',
    }});

    // ── Floating panel ────────────────────────────────────────────────────
    const panel = document.createElement('div');
    Object.assign(panel.style, {{
      background: '#1e1e2e',
      border: '1px solid #45475a',
      borderRadius: '16px',
      padding: '26px 26px 20px',
      maxWidth: '520px',
      width: 'calc(100vw - 32px)',
      maxHeight: '85vh',
      overflowY: 'auto',
      boxShadow: '0 28px 80px rgba(0,0,0,0.65)',
      display: 'flex',
      flexDirection: 'column',
      gap: '14px',
      transform: 'scale(0.92)',
      opacity: '0',
      transition: 'transform 0.22s cubic-bezier(0.34,1.56,0.64,1), opacity 0.18s ease',
    }});

    // ── Header row: question + mode badge ─────────────────────────────────
    const header = document.createElement('div');
    Object.assign(header.style, {{
      display: 'flex', alignItems: 'flex-start', gap: '12px',
    }});

    const questionEl = document.createElement('p');
    Object.assign(questionEl.style, {{
      margin: '0', color: '#cdd6f4', fontSize: '15px',
      fontWeight: '600', lineHeight: '1.55', flex: '1',
      wordBreak: 'break-word',
    }});
    questionEl.textContent = question;   // textContent = safe from XSS

    const badge = document.createElement('span');
    Object.assign(badge.style, {{
      flexShrink: '0', fontSize: '10px', fontWeight: '700',
      letterSpacing: '0.07em', padding: '3px 9px', borderRadius: '99px',
      background: accent + '26', color: accent, marginTop: '2px',
      whiteSpace: 'nowrap',
    }});
    badge.textContent = multiSelect ? 'MULTI-SELECT' : 'CHOOSE ONE';

    header.appendChild(questionEl);
    header.appendChild(badge);

    // ── Multi-select progress indicator ───────────────────────────────────
    const progressEl = document.createElement('div');
    Object.assign(progressEl.style, {{
      fontSize: '12px', color: accent, textAlign: 'right',
      minHeight: '16px', transition: 'opacity 0.15s',
      display: multiSelect ? 'block' : 'none',
    }});

    function updateProgress() {{
      if (!multiSelect) return;
      const n = selectedIndices.size;
      progressEl.textContent = n === 0
        ? 'Select one or more options'
        : n + ' of ' + options.length + ' selected';
    }}
    updateProgress();

    // ── Options ───────────────────────────────────────────────────────────
    const optContainer = document.createElement('div');
    Object.assign(optContainer.style, {{
      display: 'flex', flexDirection: 'column', gap: '7px',
    }});

    const optButtons = [];

    options.forEach(function(opt, i) {{
      const keyLabel = i < 26 ? String.fromCharCode(65 + i) : String(i + 1);
      const desc     = (descriptions[i] || '').trim();

      const btn = document.createElement('button');
      Object.assign(btn.style, {{
        display: 'flex', alignItems: 'center', gap: '11px',
        background: '#313244', border: '1.5px solid #45475a',
        borderRadius: '10px', padding: '11px 13px',
        cursor: 'pointer', textAlign: 'left', width: '100%',
        minHeight: '48px',   // accessible touch target
        transition: 'background 0.12s, border-color 0.12s, transform 0.1s',
        outline: 'none', fontFamily: 'inherit', boxSizing: 'border-box',
      }});

      // Keyboard shortcut label
      const keyBadge = document.createElement('span');
      Object.assign(keyBadge.style, {{
        flexShrink: '0', width: '26px', height: '26px',
        borderRadius: '6px', background: '#45475a',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '11px', fontWeight: '700', color: accent,
        transition: 'background 0.12s, color 0.12s', userSelect: 'none',
      }});
      keyBadge.textContent = keyLabel;

      // Text block (label + optional description)
      const textBlock = document.createElement('div');
      Object.assign(textBlock.style, {{ flex: '1', minWidth: '0' }});

      const optLabel = document.createElement('span');
      Object.assign(optLabel.style, {{
        display: 'block', color: '#cdd6f4', fontSize: '14px',
        fontWeight: '500', wordBreak: 'break-word',
      }});
      optLabel.textContent = opt;   // safe from XSS
      textBlock.appendChild(optLabel);

      if (desc) {{
        const optDesc = document.createElement('span');
        Object.assign(optDesc.style, {{
          display: 'block', color: '#a6adc8', fontSize: '12px',
          marginTop: '2px', wordBreak: 'break-word', lineHeight: '1.4',
        }});
        optDesc.textContent = desc;  // safe from XSS
        textBlock.appendChild(optDesc);
      }}

      // Checkmark indicator (multi-select only)
      const check = document.createElement('span');
      Object.assign(check.style, {{
        flexShrink: '0', fontSize: '15px', color: accent,
        opacity: '0', transition: 'opacity 0.12s',
        display: multiSelect ? 'block' : 'none',
      }});
      check.textContent = '✓';

      btn.appendChild(keyBadge);
      btn.appendChild(textBlock);
      btn.appendChild(check);

      // ── Visual state helpers ─────────────────────────────────────────
      function applySelected() {{
        btn.style.background    = accent + '1c';
        btn.style.borderColor   = accent;
        keyBadge.style.background = accent;
        keyBadge.style.color    = '#ffffff';
        check.style.opacity     = '1';
      }}
      function applyDeselected() {{
        btn.style.background    = '#313244';
        btn.style.borderColor   = '#45475a';
        keyBadge.style.background = '#45475a';
        keyBadge.style.color    = accent;
        check.style.opacity     = '0';
      }}

      btn.addEventListener('mouseenter', function() {{
        if (!selectedIndices.has(i)) {{
          btn.style.background  = '#3c3c52';
          btn.style.borderColor = accent + '77';
          btn.style.transform   = 'translateY(-1px)';
        }}
      }});
      btn.addEventListener('mouseleave', function() {{
        if (!selectedIndices.has(i)) applyDeselected();
        btn.style.transform = '';
      }});
      btn.addEventListener('focus', function() {{
        if (!selectedIndices.has(i)) btn.style.borderColor = accent + 'aa';
      }});
      btn.addEventListener('blur', function() {{
        if (!selectedIndices.has(i)) btn.style.borderColor = '#45475a';
      }});

      btn.addEventListener('click', function() {{
        if (multiSelect) {{
          if (selectedIndices.has(i)) {{
            selectedIndices.delete(i);
            applyDeselected();
          }} else {{
            selectedIndices.add(i);
            applySelected();
          }}
          updateProgress();
          // Enable/disable confirm button based on selection
          const hasAny = selectedIndices.size > 0;
          confirmBtn.style.opacity       = hasAny ? '1' : '0.38';
          confirmBtn.style.pointerEvents = hasAny ? 'auto' : 'none';
          confirmBtn.style.cursor        = hasAny ? 'pointer' : 'default';
        }} else {{
          finish({{ type: 'select', indices: [i], values: [opt] }});
        }}
      }});

      optButtons.push({{ btn, applySelected, applyDeselected }});
      optContainer.appendChild(btn);
    }});

    // ── Custom free-text row ──────────────────────────────────────────────
    if (allowCustom) {{
      const customRow = document.createElement('div');
      Object.assign(customRow.style, {{
        display: 'flex', gap: '7px', alignItems: 'stretch',
      }});

      const customInput = document.createElement('input');
      customInput.type = 'text';
      customInput.placeholder = 'Or type a custom answer\u2026';
      Object.assign(customInput.style, {{
        flex: '1', background: '#313244', border: '1.5px solid #45475a',
        borderRadius: '8px', padding: '10px 12px', color: '#cdd6f4',
        fontSize: '14px', minHeight: '44px', outline: 'none',
        fontFamily: 'inherit', transition: 'border-color 0.12s',
        boxSizing: 'border-box',
      }});
      customInput.addEventListener('focus', function() {{
        customInput.style.borderColor = accent;
        customFocused = true;
      }});
      customInput.addEventListener('blur', function() {{
        customInput.style.borderColor = customInput.value ? accent : '#45475a';
        customFocused = false;
      }});
      customInput.addEventListener('keydown', function(e) {{
        if (e.key === 'Enter' && customInput.value.trim()) {{
          e.stopPropagation();
          finish({{ type: 'custom', value: customInput.value.trim() }});
        }}
      }});

      const sendBtn = document.createElement('button');
      sendBtn.title = 'Submit custom answer (Enter)';
      Object.assign(sendBtn.style, {{
        background: accent, border: 'none', borderRadius: '8px',
        padding: '10px 16px', cursor: 'pointer', fontSize: '16px',
        color: '#ffffff', fontWeight: '700', minHeight: '44px',
        flexShrink: '0', transition: 'opacity 0.12s, transform 0.1s',
        fontFamily: 'inherit',
      }});
      sendBtn.textContent = '\u21b5';   // ↵
      sendBtn.addEventListener('mouseenter', function() {{ sendBtn.style.transform = 'scale(1.08)'; }});
      sendBtn.addEventListener('mouseleave', function() {{ sendBtn.style.transform = ''; }});
      sendBtn.addEventListener('click', function() {{
        if (customInput.value.trim()) {{
          finish({{ type: 'custom', value: customInput.value.trim() }});
        }}
      }});

      customRow.appendChild(customInput);
      customRow.appendChild(sendBtn);
      optContainer.appendChild(customRow);
    }}

    // ── Footer: Skip + Confirm ────────────────────────────────────────────
    const footer = document.createElement('div');
    Object.assign(footer.style, {{
      display: 'flex', gap: '8px', justifyContent: 'flex-end',
      marginTop: '2px',
    }});

    // Skip — hidden when required === true
    if (!required) {{
      const skipBtn = document.createElement('button');
      skipBtn.textContent = 'Skip';
      Object.assign(skipBtn.style, {{
        background: 'transparent', border: '1.5px solid #45475a',
        borderRadius: '8px', padding: '9px 18px', color: '#6c7086',
        fontSize: '13px', cursor: 'pointer', fontFamily: 'inherit',
        transition: 'border-color 0.12s, color 0.12s',
      }});
      skipBtn.addEventListener('mouseenter', function() {{
        skipBtn.style.borderColor = '#9399b2';
        skipBtn.style.color = '#cdd6f4';
      }});
      skipBtn.addEventListener('mouseleave', function() {{
        skipBtn.style.borderColor = '#45475a';
        skipBtn.style.color = '#6c7086';
      }});
      skipBtn.addEventListener('click', function() {{ finish({{ type: 'skip' }}); }});
      footer.appendChild(skipBtn);
    }}

    // Confirm — shown only in multi-select mode; starts disabled
    const confirmBtn = document.createElement('button');
    confirmBtn.textContent = 'Confirm Selection';
    Object.assign(confirmBtn.style, {{
      background: accent, border: 'none', borderRadius: '8px',
      padding: '9px 20px', color: '#ffffff', fontSize: '13px',
      fontWeight: '700', cursor: 'default', fontFamily: 'inherit',
      opacity: '0.38', pointerEvents: 'none',
      transition: 'opacity 0.15s, transform 0.1s',
      display: multiSelect ? 'block' : 'none',
    }});
    confirmBtn.addEventListener('mouseenter', function() {{
      if (selectedIndices.size > 0) confirmBtn.style.transform = 'scale(1.03)';
    }});
    confirmBtn.addEventListener('mouseleave', function() {{
      confirmBtn.style.transform = '';
    }});
    confirmBtn.addEventListener('click', function() {{
      if (selectedIndices.size === 0) return;
      const idxArr = Array.from(selectedIndices).sort(function(a,b){{return a-b;}});
      finish({{
        type: 'select',
        indices: idxArr,
        values: idxArr.map(function(i) {{ return options[i]; }}),
      }});
    }});
    footer.appendChild(confirmBtn);

    // ── Global keyboard handler ───────────────────────────────────────────
    function onKey(e) {{
      if (customFocused) return;   // let the custom input handle its own keys
      const key = e.key;

      if (key === 'Escape' && !required) {{
        finish({{ type: 'skip' }});
        return;
      }}
      if ((key === 'Enter' || key === ' ') && multiSelect && selectedIndices.size > 0) {{
        e.preventDefault();
        confirmBtn.click();
        return;
      }}
      // A–Z shortcut keys
      if (key.length === 1) {{
        const idx = key.toUpperCase().charCodeAt(0) - 65;
        if (idx >= 0 && idx < options.length) {{
          optButtons[idx].btn.click();
        }}
      }}
    }}
    document.addEventListener('keydown', onKey);

    // ── Assemble DOM ──────────────────────────────────────────────────────
    panel.appendChild(header);
    if (multiSelect) panel.appendChild(progressEl);
    panel.appendChild(optContainer);
    panel.appendChild(footer);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    // ── Entrance animation (double-rAF ensures styles are computed first) ─
    requestAnimationFrame(function() {{
      requestAnimationFrame(function() {{
        overlay.style.opacity = '1';
        panel.style.transform = 'scale(1)';
        panel.style.opacity   = '1';
      }});
    }});
  }});
}})()
"""

    def _build_rank_js(
        self,
        question: str,
        options: list[str],
        descriptions: list[str] | None = None,
        required: bool = False,
    ) -> str:
        """Build JS for a drag-to-reorder ranking overlay."""
        accent = self.valves.accent_color
        q_safe = json.dumps(question)
        opts_safe = json.dumps(options)
        descs_safe = json.dumps(descriptions or [])
        required_js = "true" if required else "false"
        accent_safe = json.dumps(accent)

        return f"""
return (function() {{
  return new Promise((resolve) => {{

    const question = {q_safe};
    const options = {opts_safe};
    const descriptions = {descs_safe};
    const required = {required_js};
    const accent = {accent_safe};
    const OVERLAY_ID = '__owui_ask_user_q__';

    const existing = document.getElementById(OVERLAY_ID);
    if (existing) existing.remove();

    // ── Current order (indices into original options array) ───────────────
    let order = options.map(function(_, i) {{ return i; }});
    let dragIdx = -1;
    let dragStartY = 0;
    let dragOffsetY = 0;
    let itemEls = [];

    function finish(payload) {{
      panel.style.transform = 'scale(0.95)';
      panel.style.opacity = '0';
      overlay.style.opacity = '0';
      setTimeout(function() {{
        overlay.remove();
        document.removeEventListener('keydown', onKey);
        resolve(JSON.stringify(payload));
      }}, 180);
    }}

    // ── Overlay ──────────────────────────────────────────────────────────
    const overlay = document.createElement('div');
    overlay.id = OVERLAY_ID;
    Object.assign(overlay.style, {{
      position: 'fixed', inset: '0', zIndex: '999999',
      background: 'rgba(0,0,0,0.62)', backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif",
      opacity: '0', transition: 'opacity 0.18s ease',
    }});

    const panel = document.createElement('div');
    Object.assign(panel.style, {{
      background: '#1e1e2e', border: '1px solid #45475a',
      borderRadius: '16px', padding: '26px 26px 20px',
      maxWidth: '520px', width: 'calc(100vw - 32px)',
      maxHeight: '85vh', overflowY: 'auto',
      boxShadow: '0 28px 80px rgba(0,0,0,0.65)',
      display: 'flex', flexDirection: 'column', gap: '14px',
      transform: 'scale(0.92)', opacity: '0',
      transition: 'transform 0.22s cubic-bezier(0.34,1.56,0.64,1), opacity 0.18s ease',
    }});

    // ── Header ───────────────────────────────────────────────────────────
    const header = document.createElement('div');
    Object.assign(header.style, {{ display: 'flex', alignItems: 'flex-start', gap: '12px' }});

    const questionEl = document.createElement('p');
    Object.assign(questionEl.style, {{
      margin: '0', color: '#cdd6f4', fontSize: '15px',
      fontWeight: '600', lineHeight: '1.55', flex: '1', wordBreak: 'break-word',
    }});
    questionEl.textContent = question;

    const badge = document.createElement('span');
    Object.assign(badge.style, {{
      flexShrink: '0', fontSize: '10px', fontWeight: '700',
      letterSpacing: '0.07em', padding: '3px 9px', borderRadius: '99px',
      background: accent + '26', color: accent, marginTop: '2px', whiteSpace: 'nowrap',
    }});
    badge.textContent = 'DRAG TO RANK';
    header.appendChild(questionEl);
    header.appendChild(badge);

    const hint = document.createElement('div');
    Object.assign(hint.style, {{
      fontSize: '12px', color: '#7f849c', marginTop: '-6px',
    }});
    hint.textContent = 'Drag items to reorder. #1 = highest priority.';

    // ── Sortable list ────────────────────────────────────────────────────
    const listContainer = document.createElement('div');
    Object.assign(listContainer.style, {{
      display: 'flex', flexDirection: 'column', gap: '6px',
      position: 'relative', userSelect: 'none', WebkitUserSelect: 'none',
    }});

    function renderList() {{
      listContainer.innerHTML = '';
      itemEls = [];

      order.forEach(function(origIdx, rank) {{
        const opt = options[origIdx];
        const desc = (descriptions[origIdx] || '').trim();

        const item = document.createElement('div');
        Object.assign(item.style, {{
          display: 'flex', alignItems: 'center', gap: '10px',
          background: '#313244', border: '1.5px solid #45475a',
          borderRadius: '10px', padding: '10px 12px',
          cursor: 'grab', minHeight: '48px',
          transition: 'transform 0.15s ease, box-shadow 0.15s ease, background 0.12s',
          touchAction: 'none', position: 'relative',
        }});
        item.dataset.rank = String(rank);

        // Rank number
        const rankBadge = document.createElement('span');
        Object.assign(rankBadge.style, {{
          flexShrink: '0', width: '28px', height: '28px',
          borderRadius: '8px', background: rank === 0 ? accent : '#45475a',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '12px', fontWeight: '800',
          color: rank === 0 ? '#ffffff' : '#a6adc8',
          transition: 'background 0.15s, color 0.15s',
        }});
        rankBadge.textContent = String(rank + 1);

        // Text
        const textBlock = document.createElement('div');
        Object.assign(textBlock.style, {{ flex: '1', minWidth: '0' }});
        const label = document.createElement('span');
        Object.assign(label.style, {{
          display: 'block', color: '#cdd6f4', fontSize: '14px',
          fontWeight: '500', wordBreak: 'break-word',
        }});
        label.textContent = opt;
        textBlock.appendChild(label);
        if (desc) {{
          const descEl = document.createElement('span');
          Object.assign(descEl.style, {{
            display: 'block', color: '#a6adc8', fontSize: '12px',
            marginTop: '2px', wordBreak: 'break-word', lineHeight: '1.4',
          }});
          descEl.textContent = desc;
          textBlock.appendChild(descEl);
        }}

        // Drag handle
        const handle = document.createElement('span');
        Object.assign(handle.style, {{
          flexShrink: '0', color: '#585b70', fontSize: '16px',
          cursor: 'grab', padding: '0 4px', lineHeight: '1',
        }});
        handle.textContent = '\\u2261';  // ≡ hamburger

        item.appendChild(rankBadge);
        item.appendChild(textBlock);
        item.appendChild(handle);

        // ── Pointer-based drag ───────────────────────────────────────
        item.addEventListener('pointerdown', function(e) {{
          if (e.button !== 0 && e.pointerType !== 'touch') return;
          e.preventDefault();
          dragIdx = rank;
          dragStartY = e.clientY;
          dragOffsetY = 0;

          item.style.zIndex = '10';
          item.style.boxShadow = '0 8px 24px rgba(0,0,0,0.4)';
          item.style.background = accent + '14';
          item.style.cursor = 'grabbing';
          item.style.transition = 'box-shadow 0.15s, background 0.12s';

          if (typeof item.setPointerCapture === 'function') {{
            item.setPointerCapture(e.pointerId);
          }}

          function onMove(ev) {{
            if (dragIdx < 0) return;
            ev.preventDefault();
            dragOffsetY = ev.clientY - dragStartY;
            item.style.transform = 'translateY(' + dragOffsetY + 'px) scale(1.02)';

            // Check if we should swap
            const itemHeight = item.getBoundingClientRect().height + 6;
            if (Math.abs(dragOffsetY) > itemHeight * 0.55) {{
              const dir = dragOffsetY > 0 ? 1 : -1;
              const newRank = dragIdx + dir;
              if (newRank >= 0 && newRank < order.length) {{
                // Swap in order array
                const tmp = order[dragIdx];
                order[dragIdx] = order[newRank];
                order[newRank] = tmp;
                dragIdx = newRank;
                dragStartY = ev.clientY;
                dragOffsetY = 0;
                renderList();
                // Re-grab the new item at newRank
                const newItem = itemEls[newRank];
                if (newItem) {{
                  newItem.style.zIndex = '10';
                  newItem.style.boxShadow = '0 8px 24px rgba(0,0,0,0.4)';
                  newItem.style.background = accent + '14';
                  newItem.style.cursor = 'grabbing';
                  newItem.style.transition = 'box-shadow 0.15s, background 0.12s';
                }}
              }}
            }}
          }}

          function onUp(ev) {{
            dragIdx = -1;
            item.style.transform = '';
            item.style.zIndex = '';
            item.style.boxShadow = '';
            item.style.background = '#313244';
            item.style.cursor = 'grab';
            item.style.transition = 'transform 0.15s ease, box-shadow 0.15s ease, background 0.12s';
            item.removeEventListener('pointermove', onMove);
            item.removeEventListener('pointerup', onUp);
            item.removeEventListener('pointercancel', onUp);
            if (typeof item.releasePointerCapture === 'function') {{
              try {{ item.releasePointerCapture(ev.pointerId); }} catch(_) {{}}
            }}
            renderList();
          }}

          item.addEventListener('pointermove', onMove);
          item.addEventListener('pointerup', onUp);
          item.addEventListener('pointercancel', onUp);
        }});

        itemEls.push(item);
        listContainer.appendChild(item);
      }});
    }}
    renderList();

    // ── Move buttons (accessible fallback for non-drag) ──────────────────
    // Keyboard: arrow keys move focused item up/down
    let focusedRank = -1;

    // ── Footer ───────────────────────────────────────────────────────────
    const footer = document.createElement('div');
    Object.assign(footer.style, {{ display: 'flex', gap: '8px', justifyContent: 'flex-end', marginTop: '2px' }});

    if (!required) {{
      const skipBtn = document.createElement('button');
      skipBtn.textContent = 'Skip';
      Object.assign(skipBtn.style, {{
        background: 'transparent', border: '1.5px solid #45475a',
        borderRadius: '8px', padding: '9px 18px', color: '#6c7086',
        fontSize: '13px', cursor: 'pointer', fontFamily: 'inherit',
        transition: 'border-color 0.12s, color 0.12s',
      }});
      skipBtn.addEventListener('mouseenter', function() {{ skipBtn.style.borderColor = '#9399b2'; skipBtn.style.color = '#cdd6f4'; }});
      skipBtn.addEventListener('mouseleave', function() {{ skipBtn.style.borderColor = '#45475a'; skipBtn.style.color = '#6c7086'; }});
      skipBtn.addEventListener('click', function() {{ finish({{ type: 'skip' }}); }});
      footer.appendChild(skipBtn);
    }}

    const confirmBtn = document.createElement('button');
    confirmBtn.textContent = 'Confirm Ranking';
    Object.assign(confirmBtn.style, {{
      background: accent, border: 'none', borderRadius: '8px',
      padding: '9px 20px', color: '#ffffff', fontSize: '13px',
      fontWeight: '700', cursor: 'pointer', fontFamily: 'inherit',
      transition: 'opacity 0.15s, transform 0.1s',
    }});
    confirmBtn.addEventListener('mouseenter', function() {{ confirmBtn.style.transform = 'scale(1.03)'; }});
    confirmBtn.addEventListener('mouseleave', function() {{ confirmBtn.style.transform = ''; }});
    confirmBtn.addEventListener('click', function() {{
      finish({{
        type: 'rank',
        indices: order.slice(),
        values: order.map(function(i) {{ return options[i]; }}),
      }});
    }});
    footer.appendChild(confirmBtn);

    // ── Keyboard ─────────────────────────────────────────────────────────
    function onKey(e) {{
      if (e.key === 'Escape' && !required) {{ finish({{ type: 'skip' }}); return; }}
      if (e.key === 'Enter') {{ confirmBtn.click(); return; }}
    }}
    document.addEventListener('keydown', onKey);

    // ── Assemble ─────────────────────────────────────────────────────────
    panel.appendChild(header);
    panel.appendChild(hint);
    panel.appendChild(listContainer);
    panel.appendChild(footer);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    requestAnimationFrame(function() {{
      requestAnimationFrame(function() {{
        overlay.style.opacity = '1';
        panel.style.transform = 'scale(1)';
        panel.style.opacity = '1';
      }});
    }});
  }});
}})()
"""

    async def ask_user_question(
        self,
        question: str,
        options: list[str],
        allow_multiple: bool = False,
        descriptions: Optional[list[str]] = None,
        required: bool = False,
        mode: str = "select",
        __event_call__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Ask the user an interactive question with clickable option buttons.

        Supports three modes:
        - "select" (default): Click one option to choose it immediately.
        - "multi_select": Click multiple options, then confirm.
        - "rank": Drag options to reorder by priority. #1 = highest.

        The user can also type a custom free-text answer (in select/multi_select modes).
        Use this whenever you need clarification, a decision, or prioritization from the user.

        :param question: The question to display (plain text, shown prominently).
        :param options: List of 2–8 options to present.
        :param allow_multiple: If true, enables multi-select mode (shorthand for mode="multi_select").
        :param descriptions: Optional subtitle text for each option (same length as options).
        :param required: If true, hides the Skip button — the user must answer.
        :param mode: "select" (pick one), "multi_select" (pick several), or "rank" (drag to reorder).
        :return: The user's selection(s), ranking, or custom answer as a human-readable string.
        """
        if not __event_call__:
            return "Error: interactive input not available in this context."

        if not options:
            return "Error: cannot ask a question with no options. Provide at least one option."

        # Normalize descriptions list to match options length
        descs: list[str] = []
        if descriptions:
            descs = [str(d) for d in descriptions[: len(options)]]
            while len(descs) < len(options):
                descs.append("")

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "⏳ Waiting for user input…",
                        "done": False,
                    },
                }
            )

        # Resolve effective mode
        effective_mode = mode.lower().strip() if mode else "select"
        if allow_multiple or self.valves.allow_multi_select:
            if effective_mode == "select":
                effective_mode = "multi_select"

        if effective_mode == "rank":
            js_code = self._build_rank_js(
                question=question,
                options=options,
                descriptions=descs or None,
                required=required,
            )
        else:
            js_code = self._build_mcq_js(
                question=question,
                options=options,
                descriptions=descs or None,
                multi_select=(effective_mode == "multi_select"),
                allow_custom=True,
                required=required,
            )

        try:
            result = await __event_call__(
                {"type": "execute", "data": {"code": js_code}}
            )
        except Exception as exc:
            logger.error("ask_user_question: execute event failed: %s", exc)
            return f"Error getting user input: {exc}"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "✅ User responded",
                        "done": True,
                        "hidden": True,
                    },
                }
            )

        if not result:
            return "User skipped the question (no answer provided)."

        # ── Parse JSON result returned by the JS overlay ──────────────────
        try:
            if isinstance(result, str):
                raw = result
            elif isinstance(result, dict):
                raw = (
                    result.get("result")
                    or result.get("value")
                    or result.get("data")
                    or ""
                )
            else:
                raw = str(result)

            parsed = json.loads(raw)
            rtype = parsed.get("type")

            if rtype == "select":
                values: list[str] = parsed.get("values", [])
                if len(values) == 1:
                    return f"User selected: {values[0]}"
                elif len(values) > 1:
                    return "User selected multiple options: " + ", ".join(values)
                else:
                    return "User submitted an empty selection."

            elif rtype == "rank":
                values = parsed.get("values", [])
                if values:
                    ranked = ", ".join(f"#{i+1} {v}" for i, v in enumerate(values))
                    return f"User's ranking (highest to lowest priority): {ranked}"
                else:
                    return "User submitted an empty ranking."

            elif rtype == "custom":
                return f"User's custom answer: {parsed.get('value', '')}"

            elif rtype == "skip":
                return "User skipped the question (no answer provided)."

            else:
                return f"User response: {raw}"

        except (json.JSONDecodeError, TypeError, AttributeError):
            raw_str = str(result).strip()
            return (
                f"User response: {raw_str}"
                if raw_str
                else "User did not provide an answer."
            )
