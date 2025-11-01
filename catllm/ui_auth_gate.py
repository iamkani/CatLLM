import streamlit as st
from typing import Optional, Sequence
from .auth import verify_user, get_user, Role, add_user, set_password, change_role, list_users, User


def _current_user() -> Optional[User]:
    auth = st.session_state.get("auth")
    if not auth:
        return None
    u = get_user(auth["username"])
    if u and u.role.value != auth.get("role"):
        st.session_state["auth"]["role"] = u.role.value
    return u


def require_login(allowed_roles: Optional[Sequence[Role]] = None) -> User:
    user = _current_user()
    if user and (allowed_roles is None or user.role in allowed_roles):
        return user

    st.title("Sign in")
    c1, c2 = st.columns(2)
    with c1:
        username = st.text_input("Username")
    with c2:
        password = st.text_input("Password", type="password")
    if st.button("Login", type="primary", use_container_width=True):
        u = verify_user(username, password)
        if not u:
            st.error("Invalid username or password.")
        elif allowed_roles and u.role not in allowed_roles:
            st.error("You do not have permission to access this page.")
        else:
            st.session_state["auth"] = {"username": u.username, "role": u.role.value}
            st.session_state["current_user"] = u.username
            st.rerun()
    st.stop()


def user_bar():
    user = _current_user()
    if not user:
        return
    with st.sidebar:
        st.caption(f"Signed in as **{user.username}** ({user.role.value})")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()


def admin_console():
    user = _current_user()
    if not user or user.role not in (Role.ADMIN, Role.ROOT):
        return
    with st.sidebar.expander("Admin Console", expanded=False):
        try:
            users_public = list_users(user)
            st.write({u: info["role"] for u, info in users_public.items()})
        except Exception as e:
            st.warning(str(e))

        st.subheader("Create user")
        nu = st.text_input("New username", key="nu")
        npw = st.text_input("New password", type="password", key="npw")
        if user.role == Role.ROOT:
            nrole = st.selectbox("Role", [Role.USER.value, Role.ADMIN.value], index=0)
        else:
            nrole = Role.USER.value
            st.caption("As Admin, you can only create User accounts.")
        if st.button("Create account"):
            try:
                newu = add_user(user, nu, npw, Role(nrole))
                st.success(f"Created {newu.username} ({newu.role.value}).")
            except Exception as e:
                st.error(str(e))

        st.subheader("Reset password")
        rp_u = st.text_input("Target username", key="rp_u")
        rp_pw = st.text_input("New password", type="password", key="rp_pw")
        if st.button("Set new password"):
            try:
                set_password(user, rp_u, rp_pw)
                st.success("Password updated.")
            except Exception as e:
                st.error(str(e))

        if user.role == Role.ROOT:
            st.subheader("Change role (Root only)")
            cr_u = st.text_input("Username to change role", key="cr_u")
            cr_role = st.selectbox("New role", [Role.USER.value, Role.ADMIN.value], index=0, key="cr_role")
            if st.button("Apply role change"):
                try:
                    change_role(user, cr_u, Role(cr_role))
                    st.success("Role updated.")
                except Exception as e:
                    st.error(str(e))
