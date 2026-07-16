"""Simple Streamlit component renderers for generative UI blocks."""

import streamlit as st

from .security import _sanitize_html, _sanitize_url


def _render_info_card(props: dict, msg_index: int = 0) -> None:
    icon = _sanitize_html(props.get("icon", "ℹ️"))
    title = _sanitize_html(props.get("title", ""))
    content = _sanitize_html(props.get("content", ""))
    color = _sanitize_html(props.get("color", "#1f77b4"))

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}15, {color}08);
            padding: 1rem 1.2rem;
            border-radius: 0.75rem;
            border-left: 4px solid {color};
            margin: 0.5rem 0;
        ">
            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.4rem;">
                {icon} {title}
            </div>
            <div style="font-size: 0.95rem; line-height: 1.5;">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_data_table(props: dict, msg_index: int = 0) -> None:
    import pandas as pd

    title = props.get("title", "")
    columns = props.get("columns", [])
    data = props.get("data", [])

    if title:
        st.markdown(f"**{_sanitize_html(title)}**")

    if columns and data:
        df = pd.DataFrame(data, columns=columns)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No data to display.")


def _render_metric(props: dict, msg_index: int = 0) -> None:
    metrics = props.get("metrics", [props])
    if not isinstance(metrics, list):
        metrics = [metrics]

    cols = st.columns(min(len(metrics), 4))
    for col, metric in zip(cols, metrics):
        with col:
            st.metric(
                label=str(metric.get("label", "")),
                value=str(metric.get("value", "")),
                delta=metric.get("delta"),
                delta_color=metric.get("delta_color", "normal"),
            )


def _render_chart(props: dict, msg_index: int = 0) -> None:
    chart_type = props.get("type", "bar")
    title = props.get("title", "")
    x = props.get("x", [])
    y = props.get("y", [])

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        series = props.get("series")

        if series:
            for s in series:
                tx, ty = s.get("x", x), s.get("y", [])
                name = s.get("name", "")
                if chart_type == "bar":
                    fig.add_trace(go.Bar(x=tx, y=ty, name=name))
                elif chart_type in ("line", "scatter"):
                    mode = "lines+markers" if chart_type == "line" else "markers"
                    fig.add_trace(go.Scatter(x=tx, y=ty, mode=mode, name=name))
        else:
            if chart_type == "bar":
                fig.add_trace(go.Bar(x=x, y=y))
            elif chart_type == "line":
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers"))
            elif chart_type == "pie":
                fig.add_trace(go.Pie(labels=x, values=y))
            elif chart_type == "scatter":
                fig.add_trace(go.Scatter(x=x, y=y, mode="markers"))

        fig.update_layout(
            title=title,
            xaxis_title=props.get("x_label", ""),
            yaxis_title=props.get("y_label", ""),
            template="plotly_dark",
            height=props.get("height", 400),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback to built-in Streamlit charts
        import pandas as pd

        if title:
            st.markdown(f"**{_sanitize_html(title)}**")
        if x and y:
            df = pd.DataFrame({"x": x, "y": y}).set_index("x")
            if chart_type == "bar":
                st.bar_chart(df)
            else:
                st.line_chart(df)
        else:
            st.info("Chart data unavailable.")


def _render_button_group(props: dict, msg_index: int = 0) -> None:
    label = props.get("label", "")
    buttons = props.get("buttons", [])

    if label:
        st.markdown(f"**{_sanitize_html(label)}**")

    if not buttons:
        return

    cols = st.columns(min(len(buttons), 4))
    for i, (col, btn) in enumerate(zip(cols, buttons)):
        with col:
            btn_text = str(btn.get("text", f"Option {i + 1}"))
            btn_value = str(btn.get("value", btn_text))
            btn_key = f"genui_btn_{msg_index}_{i}_{hash(btn_text) % 100000}"

            if st.button(btn_text, key=btn_key, use_container_width=True):
                st.session_state["genui_pending_input"] = btn_value


def _render_progress(props: dict, msg_index: int = 0) -> None:
    value = props.get("value", 0)
    label = props.get("label", "")

    if label:
        st.markdown(f"**{_sanitize_html(label)}**")
    st.progress(min(max(int(value) / 100.0, 0.0), 1.0))


def _render_alert(props: dict, msg_index: int = 0) -> None:
    level = props.get("level", "info")
    message = str(props.get("message", ""))

    dispatch = {
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
    }
    dispatch.get(level, st.info)(message)


def _render_columns(props: dict, msg_index: int = 0) -> None:
    items = props.get("items", [])
    if not items:
        return

    cols = st.columns(min(len(items), 4))
    for col, item in zip(cols, items):
        with col:
            icon = item.get("icon", "")
            title_text = item.get("title", "")
            content = item.get("content", "")
            if icon:
                st.markdown(f"### {_sanitize_html(icon)}")
            if title_text:
                st.markdown(f"**{_sanitize_html(title_text)}**")
            if content:
                st.markdown(content)


def _render_json_viewer(props: dict, msg_index: int = 0) -> None:
    title = props.get("title", "")
    data = props.get("data", {})
    expanded = props.get("expanded", True)

    if title:
        with st.expander(_sanitize_html(title), expanded=expanded):
            st.json(data)
    else:
        st.json(data)


def _render_link_cards(props: dict, msg_index: int = 0) -> None:
    links = props.get("links", [])
    if not links:
        return

    cols = st.columns(min(len(links), 3))
    for i, link in enumerate(links):
        with cols[i % 3]:
            title = _sanitize_html(link.get("title", "Link"))
            url = _sanitize_url(link.get("url", "#"))
            description = _sanitize_html(link.get("description", ""))

            st.markdown(
                f"""
                <a href="{url}" target="_blank" style="text-decoration: none;">
                    <div style="
                        background: rgba(255,255,255,0.05);
                        border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 0.5rem;
                        padding: 0.8rem;
                        margin: 0.3rem 0;
                    ">
                        <div style="font-weight: 600; color: #58a6ff;">{title}</div>
                        <div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.3rem;">
                            {description}
                        </div>
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )


def _render_steps(props: dict, msg_index: int = 0) -> None:
    steps = props.get("steps", [])
    current = props.get("current", 0)

    for i, step in enumerate(steps):
        if isinstance(step, str):
            label, desc = step, ""
        else:
            label = step.get("label", f"Step {i + 1}")
            desc = step.get("description", "")

        icon = "✅" if i < current else ("🔵" if i == current else "⬜")
        line = f"{icon} **{_sanitize_html(label)}**"
        if desc:
            line += f"  \n{_sanitize_html(desc)}"
        st.markdown(line)


def _render_tabs(props: dict, msg_index: int = 0) -> None:
    tabs_data = props.get("tabs", [])
    if not tabs_data:
        return

    tab_labels = [t.get("label", f"Tab {i + 1}") for i, t in enumerate(tabs_data)]
    tabs = st.tabs(tab_labels)

    for tab, tab_data in zip(tabs, tabs_data):
        with tab:
            st.markdown(tab_data.get("content", ""))


def _render_callout(props: dict, msg_index: int = 0) -> None:
    emoji = _sanitize_html(props.get("emoji", "💡"))
    title = _sanitize_html(props.get("title", ""))
    content = _sanitize_html(props.get("content", ""))
    color = _sanitize_html(props.get("color", "#ffd700"))

    title_html = f"<strong>{emoji} {title}</strong><br/>" if title else f"{emoji} "

    st.markdown(
        f"""
        <div style="
            background: {color}12;
            border: 1px solid {color}40;
            border-radius: 0.5rem;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
        ">
            {title_html}{content}
        </div>
        """,
        unsafe_allow_html=True,
    )
