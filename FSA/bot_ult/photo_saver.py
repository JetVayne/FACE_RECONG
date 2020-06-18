# coding: utf-8
import datetime
from bot_ult.hash_tool import HashTool
from glance.jf_ult.log_tool import LogTool


class PhotoSaver:
    SAVE_DIR = 'photo_pool'

    @staticmethod
    def download(telegram_upd, photo_index: int) -> dict:
        result = dict()
        try:
            time_token = HashTool.md5_hash(str(datetime.datetime.now()))[0:9]
            img_id = '{}_{}'.format(photo_index, time_token)
            path = '{}/{}.jpg'.format(PhotoSaver.SAVE_DIR, img_id)
            telegram_upd.message.photo[-1].get_file().download(custom_path=path)
            result['id'] = img_id
            result['path'] = path
        except Exception as e:
            print(LogTool.pp_exception(e))

        return result

    @staticmethod
    def get_time_str() -> str:
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
