import base64
import zipfile
import os
import requests
import time
from datetime import datetime
from io import BytesIO
from zeep import Client
from zeep.transports import Transport
from zeep.exceptions import Fault
from dotenv import load_dotenv

load_dotenv()

class BIPUploader:
    def __init__(self):
        self.bip_endpoint = os.getenv("BIP_ENDPOINT", "https://iabbzv-dev1.fa.ocs.oraclecloud.com")
        self.wsdl_url = os.getenv("BIP_WSDL_URL", f"{self.bip_endpoint}/xmlpserver/services/ExternalReportWSSService?WSDL")
        self.folder_path = "/Custom/SIFA Conversion"
        self.username = os.getenv("BIP_USERNAME")
        self.password = os.getenv("BIP_PASSWORD")

    def upload_data_model(self, sql_query: str, base_model_name: str = "DataModel") -> str:
        """
        Uploads a data model to BI Publisher and returns the created model name.
        
        Args:
            sql_query: SQL query for the data model
            base_model_name: Base name for the model (default: "DataModel")
        
        Returns:
            Generated model name if successful, None if failed
        """
        # Configuration
        wsdl_url = self.wsdl_url
        folder_path = self.folder_path
        username = self.username
        password = self.password
        
        timestamp = int(time.time())
        model_name = f"{base_model_name}_{timestamp}"

        try:
            # Create data model XML
            data_model_xml = f"""<?xml version='1.0' encoding='utf-8'?>
    <dataModel xmlns="http://xmlns.oracle.com/oxp/xmlp" version="2.1" 
            xmlns:xdm="http://xmlns.oracle.com/oxp/xmlp" 
            xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
            defaultDataSourceRef="Oracle BI EE">
        <dataProperties>
            <property name="include_parameters" value="true"/>
            <property name="auto_generate_layout" value="true"/>
            <property name="force_validation" value="true"/>
        </dataProperties>
        <dataSets>
            <dataSet name="{model_name}_DS" type="sql" enablePreview="true">
                <sql dataSourceRef="ApplicationDB_FSCM">
                    <![CDATA[{sql_query}]]>
                </sql>
            </dataSet>
        </dataSets>
        <output rootName="DATA_DS">
            <nodeList name="data-structure">
                <dataStructure tagName="DATA_DS">
                    <group name="G_{model_name}" source="{model_name}_DS"/>
                </dataStructure>
            </nodeList>
        </output>
        <validation>
            <status>COMPLETE</status>
            <timestamp>{timestamp}000</timestamp>
        </validation>
    </dataModel>"""

            # Create metadata
            metadata_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <metadata>
    <entries>
        <entry>
        <key><![CDATA[bip:DisplayName]]></key>
        <value><![CDATA[{model_name}]]></value>
        </entry>
        <entry>
        <key><![CDATA[bip:ValidationStatus]]></key>
        <value><![CDATA[VALIDATED]]></value>
        </entry>
        <entry>
        <key><![CDATA[bip:LastModified]]></key>
        <value><![CDATA[{datetime.fromtimestamp(timestamp).isoformat()}]]></value>
        </entry>
    </entries>
    </metadata>"""

            # Create ZIP package
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{model_name}.xdm", data_model_xml)
                zf.writestr("~metadata.meta", metadata_xml)
            
            encoded_zip = base64.b64encode(zip_buffer.getvalue()).decode()

            # SOAP integration
            session = requests.Session()
            session.auth = (username, password)
            transport = Transport(session=session)
            client = Client(wsdl=wsdl_url, transport=transport)

            client.service.uploadReportObject(
                reportObjectAbsolutePathURL=f"{folder_path}/{model_name}",
                objectType="xdmz",
                objectZippedData=encoded_zip
            )

            print(f"Success: Model '{model_name}' created")
            return model_name

        except Fault as e:
            print(f"SOAP Error: {str(e)}")
            return None
        except Exception as e:
            print(f"General Error: {str(e)}")
            return None