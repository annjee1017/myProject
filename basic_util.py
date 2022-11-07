import glob
import os
import shutil
import logging
import os
import time
import sys
from datetime import datetime
import threading

def logger_object():
    os.makedirs('./log', exist_ok=True)
    logger = logging.getLogger('log_util')
    logger.setLevel(logging.INFO)

    nowtime = datetime.now().strftime('%Y_%m_%d')
    file_handler = logging.FileHandler(f'./log/log_{nowtime}.log')
    stream_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# setting log
logger = logger_object()    

class BasicUtil():
    # def __init__(self):
    #     print('start basic util')

    def rm_database(self, is_folder, path, num):
        flist = sorted(glob.glob(f'{path}/*'), key=os.path.getctime, reverse=True)
        # print('\n\n\nnow file count ', path, len(flist))

        if len(flist) > num:
            for n in range(num, len(flist)):
                if is_folder:
                    shutil.rmtree(flist[n], ignore_errors= True)
                    # print(f'Removed old folders: {flist[n]}')
                else:
                    os.remove(flist[n])
                    # print(f'Removed old files: {flist[n]}')


    def shutdown_procedure(self):        
        text = f'PC SHUTDOWM'
        logger.info(text)
        # print(text)
        time.sleep(5)
        os.system('shutdown -r now')


    def reboot_procedure(self):
        text = f'PC REBOOT'
        logger.info(text)
        # print(text)
        time.sleep(5)
        os.system('reboot now')


    def retry_procedure(self):  
        text = f'PROGRAMM RESTARTED'
        logger.info(text)
        # print(text)
        time.sleep(1)
        python = sys.executable
        os.execl(python, python, * sys.argv)


    def exit_procedure(self, master):  
        text = f'PROGRAMM CLOSED'
        logger.info(text)
        # print(text)
        time.sleep(1)
        
        master.quit()
