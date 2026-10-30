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
    * Unlike the API endpoint, this script does NOT enforce the password policy
      (10+ chars with upper, lower, digit and symbol), so pick a strong password.
    * The password is passed as a command-line argument, so it may be visible in
      shell history and process listings. Clear it from history after use.
    * Creations here are not written to the audit log.
"""
import sys
import bcrypt
from web.app import app, db, User, ROLES

if __name__ == "__main__":
    # Expect exactly: script, username, role, password; role must be a known role.
    if len(sys.argv) != 4 or sys.argv[2] not in ROLES:
        raise SystemExit(f"usage: create_user <username> <{'|'.join(sorted(ROLES))}> <password>")
    u, role, pw = sys.argv[1:]
    # Database access needs the Flask application context.
    with app.app_context():
        if User.query.filter_by(username=u).first():
            raise SystemExit("user exists")
        db.session.add(User(username=u, role=role,
                            password_hash=bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()))
        db.session.commit()
        print("created", u, role)
