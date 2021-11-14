from pathlib import Path
from typing import Optional
from tinydb import TinyDB, Query, where
from passlib.hash import bcrypt

g: Optional[object] = None
lp: str = 'mlapi db:'


class Database:
    def __init__(self, db_globals, prompt_to_create=True):
        global g
        g = db_globals
        self.db_path: Path = Path(g.config['db_path'])
        self.db: Optional[TinyDB] = None
        self.query: Query = Query()
        self.users: Optional[TinyDB.table] = None
        if self.db_path.is_dir():
            db_file_name = f"{g.config['db_path']}/db.json"
            self.db = TinyDB(db_file_name)
            self.users = self.db.table('users')
            if not len(self.users) and prompt_to_create:
                self.create_prompt()
            elif not len(self.users) and not prompt_to_create:
                g.logger.error(f"{lp} there are no configured users in the MLAPI Database! you must create a mlapi "
                               f"DB user by running python3 ml_dbuser.py")

        else:
            print(f"{lp} the config has 'db_path' configured but the path does not exist as a directory! please check "
                  f"your configuration for spelling errors.")
            g.logger.log_close()

    @staticmethod
    def _get_hash(password):
        return bcrypt.hash(password)

    def create_prompt(self, args=None):
        if args is None:
            args = {}
        print(f"|--------- MLAPI Database ---------|")
        print("You must configure at least one user!\n")
        print("!---------------! User Creation !------------!")
        p1 = None
        p2 = True
        while True:
            name = input('user name: ')
            if not name:
                print("Error: username required!\n")
                continue
            if self.get_user(name) and not args.get('force'):
                print(f"{lp} user '{name}' already exists! you must --force or remove the user and re create\n")
                return
            p1 = input('Please enter password:')
            if not p1:
                print("Error: password required\n")
                continue
            p2 = input('Please re-enter password for confirmation:')
            if p1 != p2:
                print("Passwords do not match!\n")
                continue
            break
        if p1:
            _hash = self._get_hash(p1)
            self.users.insert(
                {
                    'name': name,
                    'password': _hash
                }
            )
            print(f"|------ SUCCESS ------|")
            print(f"------- User: {name} created pw: {p2} HASH -> {_hash}  ----------------")
            return True
        return False

    def check_credentials(self, user, supplied_password, ip=None):
        user_object = self.get_user(user)
        if ip is None:
            ip = '<Unable to obtain requesting IP>'
        if not user_object:
            g.logger.info(f"{lp} login FAILED for user -> '{user}' IP: {ip} [no such user]")
            return False  # user doesn't exist
        stored_password_hash = user_object.get('password')

        if not bcrypt.verify(supplied_password, stored_password_hash):
            # for fail2ban filter, set to INFO so we always see it
            g.logger.info(f"{lp} login FAILED for user -> '{user}' ({ip}) [incorrect password]")
            stored_password_hash, supplied_password = None, None
            return False
        else:
            g.logger.debug(f"{lp} login SUCCEEDED for user -> '{user}' ({ip}) [correct password]")
            stored_password_hash, supplied_password = None, None
            return True

    def get_all_users(self):
        return self.users.all()

    def get_user(self, user):
        return self.users.get(self.query.name == user)

    def delete_user(self, user):
        return self.users.remove(where('name') == user)

    def add_user(self, user, password):
        hashed_password = self._get_hash(password)
        return self.users.upsert(
            {
                'name': user,
                'password': hashed_password
            },
            self.query.name == user
        )
