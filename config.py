# coding :utf-8
#
#
#
#
import configparser as ConfigParser


def load_conf_info(config_file):
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    return config
