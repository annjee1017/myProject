import os
import re
import py_compile as pc
from datetime import datetime
import shutil

a = list(os.walk(os.getcwd()))
backup_folder = str(datetime.strftime(datetime.now(), '%y%m%d_%H%M%S'))
final_backup_dir = os.path.join('.backup', backup_folder)
os.makedirs(final_backup_dir, exist_ok=True)

for dumps in a:
    total_dir, folders, files = dumps
    for j in [os.path.join(total_dir, i) for i in files]:
        # continue if file is compiler
        if 'compiler.py' in j:
            continue
        if '.py' not in j:
            continue
        print('[Current file]', j)

        # compile every <.py> to <.pyc>
        kk = pc.compile(j, cfile=re.sub('.py$', '.pyc', j))
        backup_j = j.split(os.sep)[-1]
        print(backup_j)

        # backup files
        backup_folder = str(datetime.strftime(datetime.now(), '%y%m%d_%H%M%S'))
        final_backup_dir = os.path.join('.backup', backup_folder)
        os.makedirs(final_backup_dir, exist_ok=True)
        shutil.move(j, os.path.join(final_backup_dir, backup_j))
