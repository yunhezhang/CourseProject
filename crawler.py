from bs4 import BeautifulSoup
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from classificationModel import faculty_directory_classification
import re 
import urllib
import time

class crawler:

    def __init__(self):
        # Set up web driver and SVM classification model
        options = Options()
        options.headless = True
        self.driver = webdriver.Chrome('./chromedriver',options=options)
        self.max_found = 200
        self.model = faculty_directory_classification()
        self.model.train()

    def get_js_soup(self, url, driver):
        driver.get(url)
        res_html = driver.execute_script('return document.body.innerHTML')
        soup = BeautifulSoup(res_html,'html.parser') #beautiful soup object to be used for parsing html content
        return soup

    def is_valid_url(self, url):
        remove_words = ['https://', 'http://']
        hint_words = ['faculty', 'Faculty', 'people', 'People', 'staff', 'Staff', 'members', 'Members']
        for remove_word in remove_words:
            url = url.replace(remove_word, '')

        # Remove empty string
        tokens = [token for token in url.split('/') if token]

        for token in tokens:
            for hint_word in hint_words:
                if hint_word in token and (len(token) != len(hint_word)):
                    return False

        return True

    def crawl_directory_url(self, host_url):
        soup = self.get_js_soup(host_url, self.driver)
        links = soup.find_all('a')
        directory_urls = []

        for link in links:
            directory_url = host_url + link.get('href', '')
            prediction = self.model.predict(directory_url)
            if prediction == '1' and self.is_valid_url(directory_url) and (directory_url not in directory_urls):
                directory_urls.append(directory_url)

        return directory_urls

    def crawl_faculty_url(self, host_url, directory_url):
        soup = self.get_js_soup(directory_url, self.driver)
        links = soup.find_all('a')
        faculty_urls = []

        for link in links:
            faculty_url = host_url + link.get('href', '')
            prediction = self.model.predict(faculty_url)
            if prediction == '1' and (faculty_url not in faculty_urls) and (faculty_url != directory_url):
                faculty_urls.append(faculty_url)

        return faculty_urls

    def crawl(self, host_url):
        # First crawl directory urls
        directory_urls = self.crawl_directory_url(host_url)
        
        # Crawl faculty urls until reaching maximum numbers
        faculty_urls = []
        for directory_url in directory_urls:
            if (len(faculty_urls) >= self.max_found):
                break
            crawled_urls = self.crawl_faculty_url(host_url, directory_url)
            for crawled_url in crawled_urls:
                if crawled_url not in faculty_urls:
                    faculty_urls.append(crawled_url)

        return faculty_urls
            


