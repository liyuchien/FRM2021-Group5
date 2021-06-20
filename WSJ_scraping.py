#!/usr/bin/env python
# coding: utf-8

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotInteractableException
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup 
import pandas as pd
from datetime import datetime,timedelta
import time
import os 

def time_period(start, end):
    from_date = datetime.strptime(start,"%Y%m%d").date()
    to_date = datetime.strptime(end,"%Y%m%d").date()
    dates = []
    delta = to_date - from_date
    for i in range(delta.days + 1):
        dates.append(str(from_date + timedelta(days=i)).replace('-', '/'))
    return dates


def one_day_article(date):
    url = 'https://www.wsj.com/news/archive/' + date
    
    driver.get(url)
    time.sleep(1)
    
    print('=====================')
    print(date+' is processing...')

    oneday_links = []
    linkslist_ol = driver.find_element_by_xpath('//*[@id="main"]/div[1]/div/ol')
    linkslist = oneday_links.extend(list(set([i.get_attribute("href") for i in linkslist_ol.find_elements_by_tag_name('a')])))

    options = driver.find_element_by_xpath('//*[@id="main"]/div[2]')
    option_ul = options.find_element_by_tag_name('ul')
    option = len([i for i in option_ul.find_elements_by_tag_name('div')])

    for times in range(option-1):
        div = driver.find_element_by_xpath('//*[@id="main"]/div[2]')
        div.find_element_by_tag_name('a').click()
        time.sleep(1)
        linkslist_ol = driver.find_element_by_xpath('//*[@id="main"]/div[1]/div/ol')
        linkslist = oneday_links.extend(list(set([i.get_attribute("href") for i in linkslist_ol.find_elements_by_tag_name('a')])))

    oneday_article = []   
    
    i=1
    for link in oneday_links:
        
        try: 
            driver.get(link)
            time.sleep(3)
            
            article = ''
            contents = driver.find_element_by_xpath('//*[@id="wsj-article-wrap"]')
            for ele in contents.find_elements_by_tag_name('p'):
                article += ele.text
            oneday_article.append(article)
            print(str(i)+'/'+str(len(oneday_links))+' SUCCESS!')
            
        except (NoSuchElementException, StaleElementReferenceException, TimeoutException) as e:
            print(str(i)+'/'+str(len(oneday_links))+' FAIL! -- '+ str(e))
            pass
        
        i+=1

    df = pd.DataFrame([oneday_article])
    df.insert(0,'Date',date) 
    
    print(date+' finished')
    print('=====================')
    
    return df


url = 'https://www.wsj.com/news/archive'

chrome_options = Options()
# chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument('--start-maximized')
# chrome_options.add_argument('--headless')
prefs = {"profile.managed_default_content_settings.images": 2}
chrome_options.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

driver.get(url)

print('START')

sign_in_link = driver.find_element_by_link_text('Sign In')
sign_in_link.click()

username = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'username')))
password = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'password')))

user1 = os.environ.get("USER", USER)
pass1 = os.environ.get("PASS", PASS) 

soup = BeautifulSoup(driver.page_source, 'lxml')

username.send_keys(user1)
password.send_keys(pass1)

submit_button = driver.find_element_by_xpath(".//button[@type='submit'][@class='solid-button basic-login-submit']")
submit_button.click()

time.sleep(2)

print('Successfully log in.')

################ USER's INPUT #################
dates = time_period('20140410','20140411')
save_path = INPUT_YOUR_PATH_HERE
USER = YOUR_USERNAME
PASS = YOUR_PASSWORD
###############################################

file = os.path.join(save_path,'WSJ_articles.xlsx')

for date in dates:
    
    try:
        df = one_day_article(date)

        if  os.path.exists(file) == False:
            df.to_excel(file, index = False)
        else:
            df_old = pd.read_excel(file)
            df_combine = df_old.append(df, sort=True)
            df_combine.to_excel(file, index=False)
            
    except (NoSuchElementException, StaleElementReferenceException, TimeoutException, ElementNotInteractableException, ElementClickInterceptedException) as e:
        print(str(e))
        pass
    
driver.close()

print('END')
