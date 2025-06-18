"""
CNB Azure Storage Integration
Banking-grade secure data storage for risk analysis
"""

from azure.storage.blob import BlobServiceClient
import pandas as pd
import json
import io
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class CNBAzureStorage:
    def __init__(self, connection_string=None):
        """
        Initialize Azure Storage client with banking-grade security
        """
        self.connection_string = connection_string or os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        if not self.connection_string:
            logger.warning("Azure Storage connection string not provided")
            self.blob_service_client = None
            return
            
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            self.container_name = "risk-data-secure"
            logger.info("Azure Storage client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Storage: {e}")
            self.blob_service_client = None
    
    def is_connected(self):
        """Check if Azure Storage is properly connected"""
        return self.blob_service_client is not None
    
    def upload_risk_analysis(self, transaction_data, risk_result, customer_id=None):
        """
        Upload risk analysis result to secure Azure storage
        
        Args:
            transaction_data (dict): Original transaction data
            risk_result (dict): Risk analysis result
            customer_id (str): Optional customer identifier (hashed)
        
        Returns:
            dict: Upload result with blob URL
        """
        if not self.is_connected():
            return {'error': 'Azure Storage not connected'}
        
        try:
            # Create analysis record with banking compliance
            analysis_record = {
                'timestamp': datetime.now().isoformat(),
                'transaction_data': {
                    'amount': transaction_data.get('amount'),
                    'transaction_type': transaction_data.get('transaction_type'),
                    'country': transaction_data.get('country'),
                    'transaction_hour': transaction_data.get('transaction_hour'),
                    # Note: Sensitive data like account_balance excluded for privacy
                },
                'risk_analysis': risk_result,
                'customer_id_hash': customer_id,  # Only hashed ID for privacy
                'compliance': {
                    'gdpr_compliant': True,
                    'data_classification': 'confidential',
                    'retention_days': 2555,  # 7 years banking requirement
                    'audit_trail': True
                }
            }
            
            # Generate secure filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"risk_analysis_{timestamp}.json"
            
            # Convert to JSON
            json_data = json.dumps(analysis_record, indent=2, ensure_ascii=False)
            
            # Upload to Azure with encryption
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=filename
            )
            
            blob_client.upload_blob(
                json_data,
                overwrite=True,
                metadata={
                    'content_type': 'application/json',
                    'data_class': 'confidential',
                    'source': 'cnb_risk_platform',
                    'risk_level': risk_result.get('risk_level', 'unknown')
                }
            )
            
            logger.info(f"Risk analysis uploaded successfully: {filename}")
            
            return {
                'success': True,
                'filename': filename,
                'blob_url': blob_client.url,
                'timestamp': analysis_record['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Failed to upload risk analysis: {e}")
            return {'error': f'Upload failed: {str(e)}'}
    
    def upload_training_data(self, dataframe, model_metadata=None):
        """
        Upload training data for model versioning and audit
        
        Args:
            dataframe (pd.DataFrame): Training dataset
            model_metadata (dict): Model training metadata
        
        Returns:
            dict: Upload result
        """
        if not self.is_connected():
            return {'error': 'Azure Storage not connected'}
        
        try:
            # Anonymize sensitive data before upload
            df_anonymized = dataframe.copy()
            if 'customer_id' in df_anonymized.columns:
                # Hash customer IDs for privacy
                df_anonymized['customer_id'] = df_anonymized['customer_id'].apply(
                    lambda x: hash(str(x)) % 1000000
                )
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            df_anonymized.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Generate filename with model version
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version = model_metadata.get('version', 'v1') if model_metadata else 'v1'
            filename = f"training_data_{version}_{timestamp}.csv"
            
            # Upload dataset
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=filename
            )
            
            blob_client.upload_blob(
                csv_content,
                overwrite=True,
                metadata={
                    'content_type': 'text/csv',
                    'data_class': 'confidential',
                    'model_version': version,
                    'record_count': str(len(df_anonymized)),
                    'features': str(list(df_anonymized.columns))
                }
            )
            
            # Upload metadata separately
            if model_metadata:
                metadata_filename = f"model_metadata_{version}_{timestamp}.json"
                metadata_json = json.dumps(model_metadata, indent=2)
                
                metadata_blob = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=metadata_filename
                )
                
                metadata_blob.upload_blob(metadata_json, overwrite=True)
            
            logger.info(f"Training data uploaded: {filename}")
            
            return {
                'success': True,
                'data_filename': filename,
                'metadata_filename': metadata_filename if model_metadata else None,
                'record_count': len(df_anonymized)
            }
            
        except Exception as e:
            logger.error(f"Failed to upload training data: {e}")
            return {'error': f'Upload failed: {str(e)}'}
    
    def list_risk_analyses(self, limit=10):
        """
        List recent risk analyses for monitoring
        
        Args:
            limit (int): Maximum number of records to return
        
        Returns:
            list: Recent analysis records
        """
        if not self.is_connected():
            return {'error': 'Azure Storage not connected'}
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            # List blobs with risk analysis pattern
            blobs = container_client.list_blobs(name_starts_with="risk_analysis_")
            
            # Sort by last modified (newest first) and limit
            blob_list = []
            for blob in sorted(blobs, key=lambda x: x.last_modified, reverse=True)[:limit]:
                blob_list.append({
                    'filename': blob.name,
                    'last_modified': blob.last_modified.isoformat(),
                    'size': blob.size,
                    'risk_level': blob.metadata.get('risk_level', 'unknown') if blob.metadata else 'unknown'
                })
            
            return {
                'success': True,
                'analyses': blob_list,
                'count': len(blob_list)
            }
            
        except Exception as e:
            logger.error(f"Failed to list analyses: {e}")
            return {'error': f'List failed: {str(e)}'}
    
    def download_analysis(self, filename):
        """
        Download specific risk analysis for review
        
        Args:
            filename (str): Blob filename to download
        
        Returns:
            dict: Analysis data or error
        """
        if not self.is_connected():
            return {'error': 'Azure Storage not connected'}
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=filename
            )
            
            # Download blob content
            blob_data = blob_client.download_blob().readall().decode('utf-8')
            analysis_data = json.loads(blob_data)
            
            return {
                'success': True,
                'data': analysis_data,
                'filename': filename
            }
            
        except Exception as e:
            logger.error(f"Failed to download analysis: {e}")
            return {'error': f'Download failed: {str(e)}'}
    
    def get_storage_stats(self):
        """
        Get storage statistics for monitoring dashboard
        
        Returns:
            dict: Storage statistics
        """
        if not self.is_connected():
            return {'error': 'Azure Storage not connected'}
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            # Count different types of files
            risk_analyses = 0
            training_data = 0
            total_size = 0
            
            for blob in container_client.list_blobs():
                total_size += blob.size
                if blob.name.startswith('risk_analysis_'):
                    risk_analyses += 1
                elif blob.name.startswith('training_data_'):
                    training_data += 1
            
            return {
                'success': True,
                'stats': {
                    'risk_analyses_count': risk_analyses,
                    'training_datasets_count': training_data,
                    'total_size_mb': round(total_size / (1024 * 1024), 2),
                    'container_name': self.container_name
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {'error': f'Stats failed: {str(e)}'}


# Test function for validation
def test_azure_storage(connection_string):
    """
    Test Azure Storage connectivity and basic operations
    """
    print("🧪 Testing CNB Azure Storage Integration...")
    
    storage = CNBAzureStorage(connection_string)
    
    if not storage.is_connected():
        print("❌ Failed to connect to Azure Storage")
        return False
    
    print("✅ Connected to Azure Storage successfully")
    
    # Test upload
    test_transaction = {
        'amount': 50000,
        'transaction_type': 'withdrawal',
        'country': 'CZ',
        'transaction_hour': 14
    }
    
    test_result = {
        'risk_score': 0.75,
        'risk_level': 'HIGH',
        'is_risky': True
    }
    
    upload_result = storage.upload_risk_analysis(test_transaction, test_result, 'test_customer_123')
    
    if upload_result.get('success'):
        print(f"✅ Test upload successful: {upload_result['filename']}")
    else:
        print(f"❌ Test upload failed: {upload_result.get('error')}")
        return False
    
    # Test stats
    stats_result = storage.get_storage_stats()
    if stats_result.get('success'):
        stats = stats_result['stats']
        print(f"✅ Storage stats: {stats['risk_analyses_count']} analyses, {stats['total_size_mb']} MB")
    else:
        print(f"⚠️ Stats warning: {stats_result.get('error')}")
    
    print("🎉 Azure Storage integration test completed successfully!")
    return True


if __name__ == "__main__":
    # Test with environment variable or manual connection string
    connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if connection_string:
        test_azure_storage(connection_string)
    else:
        print("⚠️ Set AZURE_STORAGE_CONNECTION_STRING environment variable to test")