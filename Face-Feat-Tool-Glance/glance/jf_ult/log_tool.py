# coding: utf-8
import datetime
import os
import sys
import traceback
import random


class LogTool:
    bar = '==============================================================================='

    @staticmethod
    def gen_random_token() -> str:
        random_token = ''
        for i in range(4):
            j = random.randrange(0, 3)
            if j == 1:
                a = random.randrange(0, 10)
                random_token += str(a)
            elif j == 2:
                a = chr(random.randrange(65, 91))
                random_token += a
            else:
                a = chr(random.randrange(97, 123))
                random_token += a

        return random_token

    @staticmethod
    def create_slog_file_name() -> str:
        return '{}-{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'), LogTool.gen_random_token())

    @staticmethod
    def update_slog(path=None, msg=None, purpose=None):
        if purpose == 'a':
            # print(msg)
            with open(path, 'a') as f:
                f.write(msg + '\n')
        elif purpose == 'm':
            with open(path, 'r') as old_file:
                lines = old_file.readlines()
                with open(path, 'r+') as new_file:
                    if len(lines) == 0:
                        lines.append(msg)
                        new_file.writelines(lines)
                    else:
                        lines[-1] = msg
                        new_file.writelines(lines)

    @staticmethod
    def update_browser_log(path=None, msg=None, purpose=None):
        if not os.path.exists(path):
            LogTool.init_file(path)

        if purpose == 'a':
            with open(path, 'a') as f:
                f.write(msg + '\n')

        elif purpose == 'm':
            with open(path, 'r') as old_file:
                lines = old_file.readlines()
                with open(path, 'r+') as new_file:
                    lines[-1] = msg
                    new_file.writelines(lines)

        elif purpose == 'ab':
            with open(path, 'a') as f:
                f.write(LogTool.bar + '\n')

    @staticmethod
    def add_bar(path):
        with open(path, 'a') as f:
            f.write(LogTool.bar + '\n')

    @staticmethod
    def init_file(path):
        with open(path, 'w') as _:
            return

    @staticmethod
    def pp_exception(e):
        error_class = e.__class__.__name__
        detail = e.args[0]
        cl, exc, tb = sys.exc_info()
        lastCallStack = traceback.extract_tb(tb)[-1]
        fileName = lastCallStack[0]
        lineNum = lastCallStack[1]
        funcName = lastCallStack[2]
        err_msg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)

        return err_msg
