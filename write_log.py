from datetime import datetime
import traceback
import os

def write_log(log):
    try:
        # 텍스트 저장할 디렉토리 생성
        text_path = f"database/log"
        os.makedirs(text_path, exist_ok=True)

        new_log = log
        new_date = new_log.split(' / ')[0]

        # 텍스트 파일 유무 확인
        if os.path.isfile(f"{text_path}/{datetime.now().strftime('%Y_%m')}.txt"):
            # 현재 텍스트 불러오기
            with open(f"{text_path}/{datetime.now().strftime('%Y_%m')}.txt", "r") as file:
                lines = file.readlines()
                last_line = [i for i in lines][-1]
                last_date = last_line.split(' / ')[0]
            
            # 날짜가 다르면 이어쓰기
            if last_date != new_date:
                with open(f"{text_path}/{datetime.now().strftime('%Y_%m')}.txt", "a") as file:
                    file.writelines(new_log)
            # 날짜가 같으면 덮어쓰기
            else:
                with open(f"{text_path}/{datetime.now().strftime('%Y_%m')}.txt", "w") as file:
                    for line in lines:
                        file.write(line.replace(last_line, new_log))
        else:
            with open(f"{text_path}/{datetime.now().strftime('%Y_%m')}.txt", "w") as file:
                file.writelines(new_log)
    except:
        text = f'[MAIN] write_log {traceback.format_exc()}'
        print(text)
        # logger.error(text)
        pass

# test #--------------------------------------------------------------------------------------------
for i in range(7):
    txt = f"{datetime.now().strftime('%Y_%m_%d')} / TOTAL: {i} / OK: {i} / NG: {i}\n"
    write_log(log = txt)
