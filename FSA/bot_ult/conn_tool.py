# coding: utf-8
import configparser
import json
import os
import sys
import requests


class ConnTool:
    @staticmethod
    def get_conn_info(config_path: str) -> dict:
        result = dict()
        cfg_parser = configparser.ConfigParser()
        cfg_parser.read(config_path)

        ngrok_api = 'http://127.0.0.1:4040/api/tunnels'
        try:
            res = requests.get(ngrok_api)
        except Exception as e:
            print(e)
            print('Please Launch Ngrok First !')
            sys.exit(0)

        # print(ngrok_api)

        ngrok_info = json.loads(res.text)

        wh_url = ''
        for tunnel in ngrok_info['tunnels']:
            if tunnel['proto'] == 'https':
                wh_url = tunnel['public_url']
                break

        cfg_parser['TELEGRAM']['WEB_HOOK_URL'] = wh_url

        user_token = cfg_parser['TELEGRAM']['ACCESS_TOKEN']

        with open('bot_ult/config.ini', 'w') as configfile:
            cfg_parser.write(configfile)

        result['token'] = user_token
        result['webhook'] = wh_url

        return result

    @staticmethod
    def set_web_hook(conn_info: dict) -> str:
        access_token = conn_info['token']
        web_hook = conn_info['webhook']
        api = 'https://api.telegram.org/bot{}/setWebhook?url={}/hook'.format(access_token, web_hook)
        print(api)
        res = requests.post(api).text
        msg = json.loads(res)['description']
        print(msg)
        return msg
