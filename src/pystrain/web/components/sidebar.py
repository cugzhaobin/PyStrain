"""Reusable sidebar widgets: step indicators, navigation buttons."""

from __future__ import annotations

from typing import List

import streamlit as st


def render_step_indicator(
    steps: List[str],
    labels: List[str],
    current_step: str,
) -> None:
    """Render a visual step indicator in the sidebar.

    Completed steps show ✅, the current step is highlighted, and
    future steps are shown dimmed.
    """
    st.markdown("### 工作流步骤")
    current_idx = steps.index(current_step) if current_step in steps else 0

    for i, (step_id, label) in enumerate(zip(steps, labels)):
        if i < current_idx:
            st.markdown(f"✅ ~~{label}~~")
        elif i == current_idx:
            st.markdown(f"**→ {label}**")
        else:
            st.markdown(f"⚪ *{label}*", help="尚未到达此步骤")


def render_navigation(
    step_key: str,
    current_step: str,
    steps: List[str],
    can_advance: bool = True,
    key_suffix: str = "",
) -> None:
    """Render Back / Next buttons in the sidebar using callbacks."""
    st.divider()
    c1, c2 = st.columns(2)
    current_idx = steps.index(current_step) if current_step in steps else 0

    with c1:
        if current_idx > 0:
            prev = steps[current_idx - 1]
            st.button(
                "← 上一步",
                key=f"back_{key_suffix}",
                on_click=lambda p=prev: st.session_state.__setitem__(step_key, p),
            )

    with c2:
        if current_idx < len(steps) - 1:
            nxt = steps[current_idx + 1]
            disabled = not can_advance
            st.button(
                "下一步 →",
                key=f"next_{key_suffix}",
                disabled=disabled,
                on_click=lambda n=nxt: st.session_state.__setitem__(step_key, n),
            )


def render_reset_button(mode: str, key_suffix: str = "") -> None:
    """Render a button to reset the current mode's workflow.

    Uses a callback pattern so reset happens BEFORE the next rerun,
    avoiding timing issues with ``st.rerun()`` inside button handlers.

    Parameters
    ----------
    mode : "strain_rate" or "timeseries".
    """
    st.divider()

    def _do_reset():
        from pystrain2.web.session import reset_mode
        reset_mode(mode)

    st.button(
        "🔄 重新开始",
        key=f"reset_{key_suffix}",
        use_container_width=True,
        on_click=_do_reset,
    )
