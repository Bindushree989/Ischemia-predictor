"""Create a user:  python -m scripts.create_user <username> <role> <password>"""
import sys
import bcrypt
from web.app import app, db, User, ROLES

if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[2] not in ROLES:
        raise SystemExit(f"usage: create_user <username> <{'|'.join(sorted(ROLES))}> <password>")
    u, role, pw = sys.argv[1:]
    with app.app_context():
        if User.query.filter_by(username=u).first():
            raise SystemExit("user exists")
        db.session.add(User(username=u, role=role,
                            password_hash=bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()))
        db.session.commit()
        print("created", u, role)
