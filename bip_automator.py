import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv
import os, re
import xml.etree.ElementTree as ET
from openpyxl import Workbook

load_dotenv()

class BIPAutomator:
    def __init__(self):
        self.driver = None
        self.base_url = os.getenv("BIP_ENDPOINT", "https://iabbzv-dev1.fa.ocs.oraclecloud.com")
        self.username = os.getenv("BIP_USERNAME")
        self.password = os.getenv("BIP_PASSWORD")
        
        # Use webdriver-manager for automatic Chrome driver management
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            self.chrome_driver_path = ChromeDriverManager().install()
        except ImportError:
            # Fallback to environment variable or default path
            self.chrome_driver_path = os.getenv("CHROME_DRIVER_PATH", "chromedriver")
        
        # Use environment variable for download directory or default to user's Downloads
        self.download_dir = os.getenv("DOWNLOAD_DIR", os.path.expanduser("~/Downloads"))
        self.main_window = None

    def initialize_driver(self):
        service = Service(self.chrome_driver_path)
        options = webdriver.ChromeOptions()
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        options.add_experimental_option("prefs", prefs)
        options.add_argument("--start-maximized")
        options.add_argument("--disable-popup-blocking")
        self.driver = webdriver.Chrome(service=service, options=options)
        return WebDriverWait(self.driver, 10)

    def login(self):
        self.driver.get(f"{self.base_url}/xmlpserver/")
        wait = WebDriverWait(self.driver, 10)
        wait.until(EC.presence_of_element_located((By.ID, "userid"))).send_keys(self.username)
        time.sleep(1)
        wait.until(EC.presence_of_element_located((By.ID, "password"))).send_keys(self.password)
        time.sleep(1)
        Select(self.driver.find_element(By.ID, "Languages")).select_by_visible_text("English")
        time.sleep(2)
        self.driver.find_element(By.ID, "btnActive").click()
        time.sleep(2)

    def _full_workflow(self, model_name: str):
        try:
            self.initialize_driver()
            self.login()
            
            if self._navigate_to_model(model_name):
                time.sleep(10)
                if self._modify_query(model_name):
                    self._save_and_export(model_name)
        finally:
            self.driver.quit()

    def _navigate_to_model(self, model_name: str) -> bool:
        try:            
            #     # Add validation
            # if not re.match(r'^[\w-]{1,50}$', model_name):
            #     raise ValueError(f"Invalid model name: {model_name}")            
            
            self.driver.get(f"{self.base_url}/xmlpserver/servlet/catalog")
            time.sleep(2)

            # Navigate through folders
            WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(@onclick,'\"/Custom\"')][contains(text(),'Custom')]"))
            ).click()
            time.sleep(1)
            
            WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@onclick,\"'SIFA Conversion'\")]"))
            ).click()
            time.sleep(1)

            # Find and click edit link
            edit_xpath = f"//a[contains(@onclick,'{model_name}.xdm')][@title='Edit Data Model']"
            edit_link = WebDriverWait(self.driver, 30).until(
                EC.element_to_be_clickable((By.XPATH, edit_xpath))
            )
            self.driver.execute_script("arguments[0].click();", edit_link)
            return True
        except Exception as e:
            print(f"Navigation failed: {str(e)}")
            return False

    def _modify_query(self, model_name: str) -> bool:
        """EXACT replication of working modify_sql_query logic"""
        try:
            if len(self.driver.window_handles) > 1:
                self.driver.switch_to.window(self.driver.window_handles[1])
                self.driver.maximize_window()

            # 1. Click the header span in nested div
            header_xpath = (
                "//div[contains(@class,'dmHeaderDiv')]//"
                f"span[starts-with(@id,'title_dg-') and "
                f"contains(@title,'Data Source:sql:{model_name}_DS')]"
            )
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, header_xpath))
            ).click()
            time.sleep(1)  # Critical UI state update wait

            # 2. Click edit button using image attributes
            edit_img_xpath = (
                "//img[contains(@src,'edit_ena.png') and "
                "@title='Edit Selected Data Set']"
            )
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, edit_img_xpath))
            ).click()

            # 3. Modify textarea with exact JS execution
            textarea_xpath = (
                f"//textarea[contains(@id,'ds_sql_query_') and "
                "@class='queryarea']"
            )
            textarea = WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.XPATH, textarea_xpath))
            )
            self.driver.execute_script(
                "arguments[0].value = arguments[0].value.trim() + ' ';",
                textarea
            )

            # 4. Save changes with precise button selector
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[text()='OK' and contains(@id,'_saveButton')]")
                )
            ).click()

            print("Successfully modified SQL query in nested header structure")
            return True

        except Exception as e:
            print(f"Header modification failed: {str(e)}")
            return False



    def _save_and_export(self, model_name: str):
        try:
            WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//a[@id='mReportToolbar-command_save']"))
            ).click()
            time.sleep(10)
            
            # self.driver.get(f"{self.base_url}/xmlpserver/servlet/ScheduleView?reportPath=/Custom/SIFA+Conversion/{model_name}.xdo")
            # WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.ID, "getXMLButton")))
            
            # Data export sequence
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[div/text()='Data']"))
            ).click()
            time.sleep(2)
            
            Select(self.driver.find_element(By.ID, "_xnum")).select_by_value("200")
            time.sleep(1)
            
            WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.ID, "getXMLButton"))
            ).click()
            time.sleep(10)
            
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, "_exportXML"))
            ).click()
            time.sleep(10)
            
            self._convert_xml_to_excel(model_name)
        except Exception as e:
            print(f"Export failed: {str(e)}")



    def _convert_xml_to_excel(self, model_name: str):
        xml_path = os.path.join(self.download_dir, f"{model_name}.xml")
        
        # Wait for download
        start_time = time.time()
        while not os.path.exists(xml_path):
            time.sleep(1)
            if time.time() - start_time > 30:
                raise FileNotFoundError(f"XML missing: {xml_path}")

        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Dynamically find the correct element name with case insensitivity
        xml_element = None
        expected_prefix = f"G_{model_name.upper()}"
        
        # Check root's direct children for matching elements
        for child in root:
            if child.tag.upper().startswith(expected_prefix.upper()):
                xml_element = child.tag
                break
        
        if not xml_element:
            available_tags = {child.tag for child in root}
            raise ValueError(f"XML element matching '{expected_prefix}' not found. Available tags: {available_tags}")

        wb = Workbook()
        ws = wb.active
        
        headers = set()
        rows = []
        
        # Process all matching elements
        for report in root.findall(xml_element):
            row_dict = {}
            for child in report:
                # Handle potential XML namespace issues
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                row_dict[tag] = child.text.strip() if child.text else ''
                headers.add(tag)
            rows.append(row_dict)
        
        # Sort headers alphabetically
        headers = sorted(headers)
        
        # Write headers and rows
        ws.append(headers)
        for row in rows:
            ws.append([row.get(h, '') for h in headers])
        
        # Save Excel file
        excel_path = os.path.join(self.download_dir, f"{model_name}.xlsx")
        wb.save(excel_path)
        print(f"Successfully created Excel file: {excel_path}")

    def execute_workflow(self, model_name: str):
        """Public method to execute full workflow"""
        self._full_workflow(model_name)   

    def automate_excel_generation(self, sql_query: str) -> str:
        """
        Main method expected by excel_generator.py
        Creates a data model, uploads it, and generates Excel
        """
        try:
            # Import BIP uploader
            from bip_upload import BIPUploader
            
            # Create timestamp-based model name
            timestamp = int(time.time())
            model_name = f"DataModel_{timestamp}"
            
            # Upload data model with SQL query
            uploader = BIPUploader()
            created_model = uploader.upload_data_model(sql_query, model_name)
            
            if not created_model:
                print("Failed to upload data model")
                return None
            
            # Execute workflow to generate Excel
            self.execute_workflow(created_model)
            
            # Return path to generated Excel file
            excel_path = os.path.join(self.download_dir, f"{created_model}.xlsx")
            
            if os.path.exists(excel_path):
                print(f"Excel file generated successfully: {excel_path}")
                return excel_path
            else:
                print(f"Excel file not found at expected path: {excel_path}")
                return None
                
        except Exception as e:
            print(f"Error in automate_excel_generation: {str(e)}")
            return None


# def execute_bip_workflow(model_name: str):
#     """Public interface for workflow execution"""
#     automator = BIPAutomator()
#     automator._full_workflow(model_name)