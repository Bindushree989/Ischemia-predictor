"""CLI helper to create a user directly in the database.

Usage (run from the project root)::

    python -m scripts.create_user <username> <role> <password>

``role`` must be one of ``Administrator``, ``Clinician`` or ``ML Engineer``
(quote it in the shell if it contains a space, e.g. ``"ML Engineer"``).

Typical use is bootstrapping the first Administrator, because the
``POST /api/v1/admin/users`` endpoint itself needs an existing Administrator.

Exit behaviour:
    * prints ``created <username> <role>`` and exits 0 on success
    * exits with a usage message if the arguments are wrong or the role is invalid
    * exits with ``user exists`` if the username is already taken

Notes:
    * The password is stored as a bcrypt hash, never in plaintext.
    * Any non-empty password is accepted by the API and this script.
    * The password is passed as a command-line argument, so it may be visible in
      shell history and process listings. Clear it from history after use.
    * Creations here are not written to the audit log.
"""
import sys
import bcrypt
from web.app import app, db, User, ROLES

if __name__ == "__main__":
    args = sys.argv[1:]
    trailing = args[2:]
    flags = [arg for arg in trailing if arg.startswith("--")]
    positional = [arg for arg in trailing if not arg.startswith("--")]
    password_from_stdin = "--password-stdin" in flags
    if_not_exists = "--if-not-exists" in flags
    if (len(args) < 3 or args[1] not in ROLES or len(flags) != len(set(flags))
            or any(flag not in {"--password-stdin", "--if-not-exists"} for flag in flags)
            or len(positional) != (0 if password_from_stdin else 1)):
        raise SystemExit(
            f"usage: create_user <username> <{'|'.join(sorted(ROLES))}> <password> "
            "[--if-not-exists] | create_user <username> <role> --password-stdin [--if-not-exists]"
        )
    u, role = args[:2]
    pw = sys.stdin.readline().rstrip("\r\n") if password_from_stdin else positional[0]
    if not pw:
        raise SystemExit("password must not be empty")
    with app.app_context():
        if User.query.filter_by(username=u).first():
            if if_not_exists:
                print("user already exists; left unchanged")
                raise SystemExit(0)
            raise SystemExit("user exists")
        db.session.add(User(username=u, role=role,
                            password_hash=bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()))
        db.session.commit()
        print("created", u, role)
