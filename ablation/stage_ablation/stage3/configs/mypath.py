import os

db_root = r'/E22201107/Data'
PROJECT_ROOT_DIR = r"/E22201107/Code/stage3"

db_names = {'PASCALContext': 'PASCALContext', 'NYUD_MT': 'NYUDv2'}
db_paths = {}
for database, db_pa in db_names.items():
    db_paths[database] = os.path.join(db_root, db_pa)
