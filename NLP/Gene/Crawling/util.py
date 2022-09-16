# coding:utf-8

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
import os
import time

URL_DATE_DICT='date_dict.txt'


def get_page_content(url):
    return requests.get(url).text

def get_page_object(url):
    """
    获取页面内容
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    return soup


def get_js_page_object(url, engine='chrome'):
    """
    获取页面动态执行后的页面内容
    """
    if engine == 'phantomjs':
        driver = webdriver.PhantomJS()
    elif engine == 'chrome':
        # options = webdriver.ChromeOptions()
        # options.add_argument("start-maximized")
        # #anti detection
        # options.add_argument('--disable-blink-features=AutomationControlled')
        # options.add_experimental_option("excludeSwitches", ["enable-automation"])
        # options.add_experimental_option('useAutomationExtension', False)

        options = webdriver.chrome.options.Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')

        # driver = webdriver.Chrome(options=options, executable_path='.\chromedriver.exe')
        driver = webdriver.Chrome(options=options, executable_path='./chromedriver')

    driver.get(url)
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    return driver, soup


def get_js_page_frame_object_by_xpath(url, xpath):
    driver, soup = get_js_page_object(url)
    driver.switch_to.frame(driver.find_element_by_xpath(xpath))
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    return driver, soup

def get_content_page_object(content):
    soup = BeautifulSoup(content, 'html.parser')

    return soup

def load_url_date_data(use_default=False):
    """
    加载每个url的数据截止日期
    """
    dict = {'*': '2021-01-01'}
    if not use_default and os.path.exists(URL_DATE_DICT):
        with open(URL_DATE_DICT, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                tmp = line.strip().split('=')
                if len(tmp) == 2:
                    dict[tmp[0]] = tmp[1]
    return dict

def save_url_date_data(dict):
    """
    保存url日期的数据
    """
    with open(URL_DATE_DICT, 'w', encoding='utf-8') as f:
        for url in dict:
            f.write('%s=%s\n' % (url, dict[url]))


def get_url_date(dict, url):
    if url in dict:
        return dict[url]
    else:
        return dict['*']


def check_date_str(str1, str2):
    """
    比较日期字符串时间先后
    """
    int1 = int(str1.replace('-', ''))
    int2 = int(str2.replace('-', ''))

    if int1 >= int2:
        return True
    else:
        return False

def is_link_element_exists(driver, link_text):
    try:
        driver.find_element_by_link_text(link_text)
        return True
    except:
        return False

def is_link_element_available(driver, link_text):
    if is_link_element_exists(driver, link_text):
        elem = get_link_element(driver, link_text)
        if elem.get_attribute('class').find('disabled') == -1:
            return True
    return False

def get_link_element(driver, link_text):
    return driver.find_element_by_link_text(link_text)

def is_xpath_element_exists(driver, xpath):
    try:
        driver.find_element_by_xpath(xpath)
        return True
    except:
        return False

def is_xpath_element_available(driver, xpath):
    if is_xpath_element_exists(driver, xpath):
        elem = get_xpath_element(driver, xpath)
        if elem.get_attribute('class').find('disabled') == -1:
            return True
    return False

def get_xpath_element(driver, xpath):
    return driver.find_element_by_xpath(xpath)

def get_element_text_by_xpath(driver, xpath):
    return driver.find_element_by_xpath(xpath).text

def set_element_text_by_xpath(driver, xpath, text):
    elem = driver.find_element_by_xpath(xpath)
    elem.send_keys(text)
    elem.send_keys(Keys.ENTER)
