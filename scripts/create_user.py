"""Create a user:  python -m scripts.create_user <username> <role> <password>"""
import sys
import re
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
    if (len(pw) < 10 or not re.search(r"[a-z]", pw) or not re.search(r"[A-Z]", pw)
            or not re.search(r"\d", pw) or not re.search(r"[^\w\s]", pw)):
        raise SystemExit("password needs 10+ characters with upper, lower, digit and symbol")
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
