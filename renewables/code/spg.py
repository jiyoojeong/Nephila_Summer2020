from selenium import webdriver
import time
import sys
import re
import pandas as pd
from datetime import date
import os

# VARS
power_futures_url = "https://platform.mi.spglobal.com/web/client?auth=inherit&overridecdc=1&#markets/powerFutures?key=5647e4a2-63af-4257-b57f-5a8a1fc4cfac"
today = date.today()
# mm/dd/y
d = today.strftime("%m%d%y")
d_slash = today.strftime('%m/%d/%Y')

# SET UP CHROMEDRIVER
chrome_options = webdriver.ChromeOptions()

chrome_options.add_argument('--no-sandbox')

chrome_options.add_argument('--headless')

chrome_options.add_argument('--disable-dev-shm-usage')

chromedriver = "/Users/jiyoojeong/desktop/C/chromedriver83"

prefs = {"download.default_directory": "/Users/jiyoojeong/desktop/C/raw/forwards/spg"}

chrome_options.add_experimental_option("prefs", prefs)
browser = webdriver.Chrome(executable_path=chromedriver, options=chrome_options)

# move to power futures spg url
browser.get(power_futures_url)


# FUNCTIONS
def login():
    browser.implicitly_wait(30)
    print('username and password set')
    # inputs = browser.find_element_by_class_name('form-control')
    # print(inputs.text)
    username = browser.find_element_by_xpath(
        '/html/body/div/div/div[2]/div[2]/div[4]/div[14]/div/div[2]/div[1]/div/form/div/div/div/div[6]/div[2]/div/div[1]/div[1]/div[1]/input')
    password = browser.find_element_by_xpath(
        '/html/body/div/div/div[2]/div[2]/div[4]/div[14]/div/div[2]/div[1]/div/form/div/div/div/div[6]/div[2]/div/div[1]/div[1]/div[2]/div/input')

    username.send_keys("jjeong@nephilaadvisors.com")
    password.send_keys("Poobearluv.1")

    sign_in = browser.find_element_by_xpath(
        '/html/body/div/div/div[2]/div[2]/div[4]/div[14]/div/div[2]/div[1]/div/form/div/div/div/div[6]/div[2]/div/div[1]/div[5]/button')
    sign_in.click()

    print('signed in.')
    browser.implicitly_wait(60)
    print('forward prices loaded.')


def filter_options():
    # FILTER TOGGLE
    browser.find_element_by_css_selector(
        '#section_1_control_19 > div.snl-hui-filter-container.hui-hideonhtmlexport > div').click()
    time.sleep(5)


def reset_browser_level():
    browser.get(power_futures_url)
    time.sleep(10)


def dictify(objects):
    '''changes a list of selenium objects into a dictionary with keys encoded by the filter name.
       ------
       params: objects - list of selenium objects
       ------
       output: d - dictionary of key(unicode string) and value(selenium object)'''
    d = {}
    for o in objects:
        d[o.text] = o
    return d


def regions(print_keys=True):
    '''finds and scrapes the regional filter buttons of the data.
       -----
       params: print_keys(bool) - if print_keys is True, it will print the keys of the regions. Default is True.
       -----
       output: market_regions (dict) - dictionary of key(region name) and value(selenium object to select filter)
       '''
    # reset_browser_level()
    filter_options()

    region_toggle = browser.find_element_by_xpath('//*[@id="section_1_control_23"]/div/div/label/div/button')
    region_toggle.click()

    print('finding all regions.')
    all_regions = browser.find_elements_by_css_selector(
        '#section_1_control_23 > div > div > label > div > div > ul > li > a')

    market_regions = dictify(all_regions)

    if print_keys:
        print('=== market regions ===')
        print(market_regions.keys())
        print('======================')

    return market_regions


def peaks(print_keys=True):
    '''finds and scrapes the regional filter buttons of the data.
       -----
       params: print_keys(bool) - if print_keys is True, it will print the keys of the regions. Default is True.
       -----
       output: on_off (dict) - dictionary of key(off or on peak) and value(selenium object to select filter)
       '''
    # PEAK TOGGLE

    # peak_toggle = browser.find_element_by_css_selector('#section_1_control_27 > div > div > label > div > button')

    peak_toggle = browser.find_element_by_xpath('//*[@id="section_1_control_27"]/div/div/label/div/button')
    peak_toggle.click()

    print('got to peak toggle')

    peak_options = browser.find_elements_by_css_selector(
        '#section_1_control_27 > div > div > label > div > div > ul > li')

    print('got to peak options')

    on_off = dictify(peak_options)

    if print_keys:
        print('=== peaks ===')
        print(on_off.keys())
        print('=============')

    return on_off


def fwd_periods(print_keys=True):
    '''
       finds and scrapes the regional filter buttons of the data.
       -----
       params: print_keys(bool) - if print_keys is True, it will print the keys of the regions. Default is True.
       -----
       output: on_off (dict) - dictionary of key(off or on peak) and value(selenium object to select filter)
    '''
    # PERIOD TOGGLE
    reset_browser_level()
    filter_options()

    period_toggle = browser.find_element_by_css_selector('#section_1_control_25 > div > div > label > div > button')
    period_toggle.click()

    # object search
    forward_terms = browser.find_elements_by_css_selector(
        '#section_1_control_25 > div > div > label > div > div > ul > li > a')
    forward_periods = dictify(forward_terms)

    if print_keys:
        print('=== forward terms ===')
        print(forward_periods.keys())
        print('=====================')

    return forward_periods


def set_start_date(date):
    period_toggle = browser.find_element_by_css_selector('#section_1_control_25 > div > div > label > div > button')
    period_toggle.click()

    # object search
    forward_terms = browser.find_elements_by_css_selector(
        '#section_1_control_25 > div > div > label > div > div > ul > li > a')
    forward_periods = dictify(forward_terms)

    return forward_periods


def export_xls_data():
    # point clicker to section-1
    browser.find_element_by_xpath('/html/body')

    browser.find_element_by_xpath(
        '/html/body/div[1]/div/div[2]/div[2]/div[4]/div[15]/div/div/div/div/div/div/div[1]/div[3]/div/div[2]/div/div[1]/div/div/nav/div/div/ul/li[10]/a').click()

    export_excel_data = browser.find_element_by_css_selector(
        '#section_1_control_8 > div > div > nav > div > div > ul > li.dropdown.snl-last-dropdown.snl-closed.open > ul > li:nth-child(8) > a')
    export_excel_data.click()
    # ideally this would save somewhere on this cloud and then i can parse it with pandas and then save to blob ????


def apply_filters():
    browser.find_element_by_xpath(
        '/html/body/div[1]/div/div[2]/div[2]/div[4]/div[15]/div/div/div/div/div/div/div[1]/div[7]/div[1]/div[2]/div[1]/div/div[4]/div[2]/button[1]').click()
    time.sleep(20)


def open_menu(s):
    browser.implicitly_wait(2)
    if s == 'region':
        toggle = browser.find_element_by_xpath(
            '/html/body/div[1]/div/div[2]/div[2]/div[4]/div[15]/div/div/div/div/div/div/div[1]/div[7]/div[1]/div[2]/div[1]/div/div[2]/div[2]/div[2]/div/div/label/div/button')
    elif s == 'peak':
        toggle = browser.find_element_by_xpath('//*[@id="section_1_control_27"]/div/div/label/div/button')
        toggle = browser.find_element_by_xpath(
            '/html/body/div[1]/div/div[2]/div[2]/div[4]/div[15]/div/div/div/div/div/div/div[1]/div[7]/div[1]/div[2]/div[1]/div/div[2]/div[2]/div[4]/div/div/label/div/button')
    elif s == 'period':
        toggle = browser.find_element_by_css_selector('#section_1_control_25 > div > div > label > div > button')
    elif s == 'date':
        toggle = browser.find_element_by_css_selector('')
    toggle.click()
    time.sleep(3)


def scrape():
    login()
    market_regions = regions()
    peaky = peaks()

    reset_browser_level()
    filenames = []
    filepaths = []
    redo = []

    date_range = pd.date_range(start='2/01/2012', periods=365 * 9).tolist()

    date_range = [dd.strftime('%m/%d/%Y') for dd in date_range]

    region_count = 0

    path = '/Users/jiyoojeong/desktop/C/raw/forwards/spg/'
    dir_list = os.listdir(path)

    for region in market_regions.keys():
        peak_count = 0
        region_count += 1
        reset_browser_level()

        for peak_type in peaky.keys():
            peak_count += 1
            print(region_count, peak_count)
            try:
                reset_browser_level()
                filter_options()
                # set date to today
                print('open filter')
                time.sleep(1)

                # open_menu('region')
                region_toggle = browser.find_element_by_xpath(
                    '//*[@id="section_1_control_23"]/div/div/label/div/button')
                region_toggle.click()
                time.sleep(1)
                print('finding all regions.')
                reg = browser.find_element_by_css_selector(
                    '#section_1_control_23 > div > div > label > div > div > ul > li:nth-child({}) > a'.format(
                        region_count))
                rt = reg.text

                # ============= only look at ERCOT market for now
                if rt != 'ERCOT':
                    print(rt, peak_count, ' not ERCOT')
                    continue
                else:
                    print(rt, peak_count)
                    reg.click()

                    print('set region')
                    time.sleep(1)

                    # open_menu('peak')
                    peak_toggle = browser.find_element_by_xpath('//*[@id="section_1_control_27"]/div/div/label/div/button')
                    peak_toggle.click()
                    time.sleep(1)

                    peak = browser.find_element_by_css_selector(
                        '#section_1_control_27 > div > div > label > div > div > ul > li:nth-child({}) > a'.format(
                            peak_count))

                    # peaky[peak_type].click()
                    peak.click()
                    print('set peak')

                    for dd in date_range:
                        # select filters
                        try:
                            filename = re.sub(r'[\W\s]+', '', rt) + '_' + str(peak_count) + '_' + re.sub(r'/', '', dd)

                            if filename + '.csv' in dir_list:
                                print('filename: ' + filename + ' already downloaded')
                                continue
                            else:
                                filter_options()
                                time.sleep(1)
                                browser.find_element_by_css_selector('input#section_1_control_31_startdate').clear()
                                browser.find_element_by_css_selector('input#section_1_control_31_startdate').send_keys(dd)
                                print('set date')

                                # apply filters
                                apply_filters()
                                print('filters applied')
                                # export

                                export_xls_data()
                                print('waiting for xlsx to download...')

                                # ----- print download progress -----
                                toolbar_width = 30

                                # setup toolbar
                                sys.stdout.flush()
                                sys.stdout.write("\b[")  # starts
                                for i in range(0, toolbar_width):
                                    time.sleep(1)
                                    # update the bar
                                    sys.stdout.write("-")
                                    sys.stdout.flush()

                                sys.stdout.write("]\n")  # this ends the progress bar

                                # convert to renamed csv file
                                filenames.append(filename)
                                filepath = path + filename
                                filepaths.append(filepath)

                                # store filename for easier access.
                                filenames.append(filepath)
                                xls = pd.ExcelFile('/Users/jiyoojeong/desktop/C/raw/forwards/spg/SPGlobalofficeworkbook.xls')
                                print('xls')
                                temp_meta = pd.read_excel(xls, 'Chart')
                                temp_data = pd.read_excel(xls, 'Data')
                                # print(temp_data)

                                print('saving ' + filepath + ' META and DATA...')
                                temp_meta.to_csv(filepath + 'METADATA.csv')
                                print('metadata done')
                                temp_data.to_csv(filepath + '.csv')
                                print('data done')

                                print('files for: ' + filepath + ' complete!\n')
                            os.remove('/Users/jiyoojeong/desktop/C/raw/forwards/spg/SPGlobalofficeworkbook.xls')

                        except Exception as e:
                            print(e.message)
                            redo.append([region, peak_type, dd])
                            print('moving onto next.')

            except Exception as e:
                print(e.message)


    print('export complete.')
    # get all databases
    dfs = {}
    for files in filenames:
        print(files.split('_'))

    print('need to redo:')
    print(redo)
    return filenames


filenames = scrape()

print(filenames)


def get_file(filename):
    print('haha')



