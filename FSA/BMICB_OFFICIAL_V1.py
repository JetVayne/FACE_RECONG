# coding: utf-8
import os
import re

import telegram
from flask import Flask, request
from telegram import ReplyKeyboardRemove, ReplyKeyboardMarkup
from telegram.ext import CommandHandler, MessageHandler, Filters, ConversationHandler
from telegram.ext import Dispatcher

from bot_ult.conn_tool import ConnTool
from bot_ult.face_validator import FaceValidator
from bot_ult.photo_saver import PhotoSaver
from bot_ult.yrp_validator import YrpValidator
from glance.jf_ult.log_tool import LogTool
from glance.face_analyst import FaceAnalyst
from sff import FEAT_A, FEAT_B
import joblib
import numpy as np
import cv2
import pandas as pd
from pprint import pprint as pp

# create img pool dir
photo_save_path = 'photo_pool'
if not os.path.exists(photo_save_path):
    os.mkdir(photo_save_path)

# create img pool dir
fg_save_path = 'fg_pool'
if not os.path.exists(fg_save_path):
    os.mkdir(fg_save_path)

# connection preparation
conn_info = ConnTool.get_conn_info('bot_ult/config.ini')
access_token = conn_info['token']
app = Flask(__name__)
bot = telegram.Bot(token=access_token)

# record
photo_index = 0

SR0_FINISH = 'SR0_FINISH'

SR1_ONE = 'SR1_ONE'
SR1_TWO = 'SR1_TWO'
SR1_THREE = 'SR1_THREE'
SR1_FOUR = 'SR1_FOUR'
SR1_FIVE = 'SR1_FIVE'
SR1_SIX = 'SR1_SIX'

SR1_FINISH = 'SR1_FINISH'
WAIT_NEW_PHOTO = 'WAIT_NEW_PHOTO'

CHOOSING, TYPING_REPLY, TYPING_CHOICE = range(3)

DATA = dict()
BMI_DATA = dict()

AUTH = ['cathaylife']
USER = ['root']


@app.route('/hook', methods=['POST'])
def webhook_handler():
    if request.method == 'POST':
        raw_res = request.get_json(force=True)
        update = telegram.Update.de_json(raw_res, bot)
        dispatcher.process_update(update)

    return 'ok'

@app.route('/data', methods=['GET'])
def display_data():
    user = request.args.get('user')
    auth = request.args.get('auth')

    if user in USER and auth in AUTH:
        query = request.args.get('q')
        data = BMI_DATA.get(query)
        if data:
            did = data.get('id')
            name = data.get('name')
            gender = data.get('gender')
            age = data.get('age')
            bmi = data.get('bmi')
            path = data.get('path')

            content = \
            '''
            <h3>id: {}<h3>
            <h3>name: {}<h3>
            <h3>gender: {}<h3> 
            <h3>age: {}<h3>
            <h3>bmi: {}<h3>
            <h3>path: {}<h3>
            '''.format(did, name, gender, age, bmi, path)

            return content

        else:
            return 'The id of data does not exist.'

    else:
        return 'Invalid user or auth-key.'




# ----------------------------------------------- Work Place -----------------------------------------------
def sr0(bot, update):
    msg = \
        '''
    這是一套保障用戶權益並提升客戶福利的安全驗證服務!
    
    1. 您可以自由選擇是否使用這套
       服務，不論是否使用都不會影
       響原本保單的權益，但使用本
       服務者，可以擁有更多額外的
       回饋哦！
    
    2. 為了確保買賣保單的雙方都能
       放心，我們使用拍照留存的方
       式，來確認用戶的投保意願是
       否明確。
    
    3. 資料均透過公司進行加密，未
       經您的同意不會進行其他利用
       您若想更改或移除照片，可以
       直接告知公司進行變更，存檔
       照片的使用權仍掌握在您手上
    
        '''
    update.message.reply_text(msg, reply_markup=ReplyKeyboardRemove())

    return re_ask(bot, update)


def is_valid_age(s: str) -> bool:
    p = r'^((1[0-5])|[1-9])?\d$'
    if re.match(p, s):
        return True
    else:
        return False


def init_data(cid: str):
    global DATA
    record = dict()
    record['name'] = None
    record['gender'] = None
    record['age'] = None
    record['agree'] = False
    record['img_path'] = None
    DATA[cid] = record


def remove_data(cid: str):
    global DATA
    del DATA[cid]


def start(bot, update: telegram.update.Update):
    cid = update.message.chat.id

    reply_keyboard = [
        ['使用國泰保單數位認證', '我想了解國泰保單數位認證']
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    # markup = ReplyKeyboardMarkup(reply_keyboard)

    msg = '您好，我是 Cathy\n您的保單數位認證助手\n有甚麼我能為您服務的嗎?'

    update.message.reply_text(msg, reply_markup=markup, resize_keyboard=True)

    return CHOOSING


def sr1_ONE(bot, update):
    cid = update.message.chat.id
    reply_keyboard = [
        ['男', '女']
    ]
    msg = '請選擇您的性別'
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

    update.message.reply_text(msg, reply_markup=markup)
    init_data(cid)

    return SR1_TWO


def sr1_TWO(bot, update):
    cid = update.message.chat.id
    gender = update.message.text
    DATA[cid]['gender'] = gender

    print('get gender: ', gender)

    msg = '請輸入您的姓名'
    update.message.reply_text(msg, reply_markup=ReplyKeyboardRemove())

    return SR1_THREE


def sr1_THREE(bot, update):
    cid = update.message.chat.id
    name = update.message.text
    DATA[cid]['name'] = name

    print('get name: ', name)

    if DATA[cid]['gender'] == '男':
        msg = '{}先生，請輸入您的年齡'.format(name)

    else:
        msg = '{}女士，請輸入您的年齡'.format(name)

    update.message.reply_text(msg, reply_markup=ReplyKeyboardRemove())

    return SR1_FOUR


def retype_age(bot, update):
    msg = '請重新輸入正確的年齡數字'
    update.message.reply_text(msg, reply_markup=ReplyKeyboardRemove())

    return SR1_FOUR


def sr1_FOUR(bot, update):
    cid = update.message.chat.id
    age = update.message.text

    if not is_valid_age(age):
        return retype_age(bot, update)

    if DATA[cid]['age']:
        age = DATA[cid]['age']
    else:
        age = update.message.text
        DATA[cid]['age'] = age

    record = DATA[cid]
    name = record['name']
    gender = record['gender']
    age = record['age']

    reply_keyboard = [
        ['正確', '重新輸入']
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)
    msg = '請確認輸入資料是否正確:\n\n姓名: {}\n\n性別: {}\n\n年齡: {}\n\n'.format(name, gender, age)
    update.message.reply_text(msg, reply_markup=markup)

    return SR1_FIVE


def sr1_FIVE(bot, update):
    cid = update.message.chat.id
    name = DATA[cid]['name']
    gender = DATA[cid]['gender']
    if gender == '男':
        gender = '先生'
    else:
        gender = '女士'

    reply_keyboard = [
        ['同意', '不同意']
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

    msg = '親愛的保戶 - {}{}，\n\n' \
          '謝謝您選擇本公司，為了確保您是本人同意這份保單的內容，\n\n' \
          '我們將會以拍照的方式進行留存，照片資料只會保存在公司的保戶資料中，\n\n' \
          '未經您的許可不會進行其他的運用，請問您是否同意使用拍照留存?'.format(name, gender)

    update.message.reply_text(msg, reply_markup=markup)

    return SR1_SIX

def disagree(bot, update):
    msg = '好的我知道了\n之前輸入之資料將不會保存，\n請您放心!'
    update.message.reply_text(msg, reply_markup=ReplyKeyboardRemove())

    return re_ask(bot, update)


def sr1_SIX(bot, update):
    msg = '正臉資料拍攝與上傳流程如下:\n\n1. 點擊左下角的📎按鈕\n\n2. 點擊📷按鈕進行拍攝\n\n3. 拍攝完畢後，點擊右下角的↑按鈕'
    update.message.reply_text(msg, reply_markup=ReplyKeyboardRemove())
    update.poll_answer()

    return sr1_SIX


def ask_for_wph(bot, update):
    print('ask if re upload photo')
    reply_keyboard = [
        ['我要重新上傳', '取消本次認證作業']
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)

    msg = '要再重新拍照或上傳新照片嗎?'

    update.message.reply_text(msg, reply_markup=markup)
    # update.poll_answer()

    return WAIT_NEW_PHOTO


def wait_for_photo(bot, update):
    print('WAIT FOR NEW PHOTO')
    msg = '好的沒問題!'
    update.message.reply_text(msg, reply_markup=ReplyKeyboardRemove())

    msg = '正臉資料拍攝與上傳流程如下:\n\n1. 點擊左下角的📎按鈕\n\n2. 點擊📷按鈕進行拍攝\n\n3. 拍攝完畢後，點擊右下角的↑按鈕'
    update.message.reply_text(msg, reply_markup=ReplyKeyboardRemove())

    update.poll_answer()
    return SR1_SIX


def sr1_SEVEN(bot, update):
    global photo_index
    photo_index += 1
    # Stage 1 : download  to img pool
    photo_info = PhotoSaver.download(update, photo_index)
    if photo_info:
        update.message.reply_text('圖片存取成功，請稍候 ...', reply_markup=ReplyKeyboardRemove())

    else:
        update.message.reply_text('圖片無法下載!')
        return ask_for_wph(bot, update)
        # return sr1_SEVEN(bot, update) # 9527

    # start Stage 2 、 Stage 3
    photo_check, photo = photo_validation(photo_info)
    if photo_check['result']:
        update.message.reply_text('人物偵測完成')
    else:
        update.message.reply_text(photo_check['msg'])
        return ask_for_wph(bot, update)

    # Stage 4 : process photo
    update.message.reply_text('分析臉部特徵中，請稍候 ...')

    predict_result = None
    try:
        predict_result = process_photo(photo, photo_info)
    except Exception as e:
        print(LogTool.pp_exception(e))
        update.message.reply_text('系統超載，或無法順利擷取臉部區塊')
        return ask_for_wph(bot, update)

    if predict_result['result']:
        ins_id = save_record(update, photo_info, predict_result['y'])

        msg = '資料編號:\n/{}\n已登錄成功，感謝您使用本服務!'.format(ins_id)
        update.message.reply_text(msg, reply_markup=ReplyKeyboardRemove())

        print(BMI_DATA[ins_id])

        return re_ask(bot, update)

    else:
        update.message.reply_text('系統超載，或無法順利擷取臉部區塊')
        return ask_for_wph(bot, update)


def re_ask(bot, update):
    print('!!! RE ASK')
    reply_keyboard = [
        ['使用國泰保單數位認證', '我想了解國泰保單數位認證']
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    msg = '還有什麼我能幫忙的嗎?'
    update.message.reply_text(msg, reply_markup=markup, resize_keyboard=True)

    return CHOOSING


def process_photo(photo, photo_info: dict) -> dict:
    result = dict()
    img_path = photo_info.get('path')
    fa = FaceAnalyst(img_path)
    fa.analyze()
    if fa.result:
        record = list()
        FEAT = FEAT_B
        for f in FEAT:
            record.append(fa.features.get(f))

        X = pd.DataFrame(columns=FEAT)
        X.loc[0] = record
        predict_bmi = SVR_MODEL.predict(X)[0]
        result['y'] = predict_bmi
        result['result'] = True

    else:
        result['result'] = False

    return result


def photo_validation(photo_info: dict) -> tuple:
    result = dict()
    photo_path = photo_info['path']

    if os.path.exists(photo_path):
        photo = cv2.imread(photo_path)

        # Stage 2 : check if a clear face is in the
        face_vad = FaceValidator(photo)
        if not face_vad.result:
            result['result'] = False
            result['msg'] = face_vad.msg
            return result, photo

        # Stage 3 : check if the face is almost front
        yrp_vad = YrpValidator(photo)
        if not yrp_vad.result:
            result['result'] = False
            result['msg'] = yrp_vad.msg
            return result, photo

        result['result'] = True
        return result, photo

    else:
        result['result'] = False
        result['msg'] = '系統無法讀取該圖片!'
        return result, None


def save_record(update, photo_info: dict, predict_bmi: float) -> str:
    cid = update.message.chat.id

    ins_id = photo_info['id']
    name = DATA[cid]['name']
    gender = DATA[cid]['gender']
    age = DATA[cid]['age']
    bmi = predict_bmi
    path = photo_info['path']

    record = dict()
    record['id'] = ins_id
    record['name'] = name
    record['gender'] = gender
    record['age'] = age
    record['bmi'] = bmi
    record['path'] = path

    BMI_DATA[ins_id] = record

    return ins_id


def done(bot, update):
    msg = '感謝您的使用，祝您順心!\n\n輸入 /start 即可再使用本服務'
    update.message.reply_text(msg, reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


CONV_handler = ConversationHandler(
    entry_points=[CommandHandler('start', start)],

    states={
        CHOOSING: [
            MessageHandler(Filters.regex('^使用國泰保單數位認證$'),
                           sr1_ONE),

            MessageHandler(Filters.regex('^我想了解國泰保單數位認證$'),
                           sr0)
        ],

        SR1_TWO: [
            MessageHandler(Filters.regex('^男|女$'),
                           sr1_TWO),
        ],

        SR1_THREE: [
            MessageHandler(Filters.text,
                           sr1_THREE),
        ],

        SR1_FOUR: [
            MessageHandler(Filters.text,
                           sr1_FOUR),
        ],

        SR1_FIVE: [
            MessageHandler(Filters.regex('^正確$'),
                           sr1_FIVE),

            MessageHandler(Filters.regex('^重新輸入$'),
                           sr1_ONE),
        ],

        SR1_SIX: [
            MessageHandler(Filters.regex('^同意$'),
                           sr1_SIX),

            MessageHandler(Filters.regex('^不同意$'),
                           disagree),

            MessageHandler(Filters.photo,
                           sr1_SEVEN),
        ],

        WAIT_NEW_PHOTO: [
            MessageHandler(Filters.photo,
                           sr1_SEVEN),

            MessageHandler(Filters.regex('^取消本次認證作業$'),
                           re_ask),

            MessageHandler(Filters.regex('^我要重新上傳'),
                           wait_for_photo),
        ],

    },

    fallbacks=[CommandHandler('EXITBMI', done)]
)

dispatcher = Dispatcher(bot, None)
dispatcher.add_handler(CONV_handler)

if __name__ == '__main__':
    SVR_MD_PATH = 'glance/models/bmi_svr_b.pkl'
    SVR_MODEL = joblib.load(SVR_MD_PATH)

    ConnTool.set_web_hook(conn_info)
    # app.run(port=7777)
    app.run(host='0.0.0.0', port=7777)
    # app.run(host='localhost', port=7777)
