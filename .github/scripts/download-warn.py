import requests
import pandas as pd
import os
from datetime import datetime
import json
import sys
from pathlib import Path

def get_california_warn_url():
    """Return the URL for California WARN data."""
    return 'https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn_report1.xlsx'

def create_directory_structure(base_dir):
    """
    Create the directory structure for downloads and processed files.
    Returns a dictionary with the paths.
    """
    current_year = str(datetime.now().year)
    
    # Create download directory
    download_dir = os.path.join(base_dir, 'downloads', 'ca', current_year)
    os.makedirs(download_dir, exist_ok=True)
    
    # Create processed directory
    processed_dir = os.path.join(base_dir, 'processed', 'ca', current_year)
    os.makedirs(processed_dir, exist_ok=True)
    
    return {
        'download_dir': download_dir,
        'processed_dir': processed_dir,
        'year': current_year
    }

def download_file(url, download_dir):
    """
    Download the Excel file from California EDD website.
    Returns the path to the downloaded file.
    """
    current_date = datetime.now().strftime('%Y%m%d')
    filename = f'ca-warn-notice-{current_date}.xlsx'
    output_path = os.path.join(download_dir, filename)
    
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    return output_path

def process_california_warn(file_path):
    """
    Process California WARN Excel file:
    - Read the 'Detailed WARN Report ' worksheet
    - Skip the first row (non-header content)
    - Clean and standardize the data
    """
    df = pd.read_excel(
        file_path, 
        sheet_name='Detailed WARN Report ',
        header=1  # Use second row as headers
    )
    
    # Add state and download timestamp
    df['state'] = 'california'
    df['download_date'] = datetime.now().isoformat()
    
    return df

def save_processed_files(df, processed_dir, year):
    """
    Save DataFrame in both CSV and Parquet formats.
    Returns a dictionary with file information.
    """
    file_info = {}
    base_name = f'ca-{year}-warn-notice'
    
    # Save as CSV
    csv_path = os.path.join(processed_dir, f'{base_name}.csv')
    df.to_csv(csv_path, index=False)
    file_info['csv'] = {
        'path': csv_path,
        'size_bytes': os.path.getsize(csv_path)
    }
    
    # Save as Parquet
    parquet_path = os.path.join(processed_dir, f'{base_name}.parquet')
    df.to_parquet(parquet_path, index=False)
    file_info['parquet'] = {
        'path': parquet_path,
        'size_bytes': os.path.getsize(parquet_path)
    }
    
    return file_info

def main():
    # Get base directory from command line argument
    if len(sys.argv) != 2:
        print("Usage: python download_warn.py BASE_DIR")
        sys.exit(1)
        
    base_dir = sys.argv[1]
    
    # Create directory structure
    dirs = create_directory_structure(base_dir)
    
    # Dictionary to store metadata
    metadata = {
        'last_updated': datetime.now().isoformat(),
        'year': dirs['year'],
        'states_processed': []
    }
    
    try:
        # Download the file
        print("Downloading California WARN data...")
        url = get_california_warn_url()
        download_path = download_file(url, dirs['download_dir'])
        
        # Process the file
        print("Processing California WARN data...")
        df = process_california_warn(download_path)
        
        # Save processed files
        print("Saving processed data...")
        file_info = save_processed_files(df, dirs['processed_dir'], dirs['year'])
        
        metadata['states_processed'].append({
            'state': 'california',
            'status': 'success',
            'records': len(df),
            'source_url': url,
            'columns': df.columns.tolist(),
            'files': {
                'download': {
                    'path': download_path,
                    'size_bytes': os.path.getsize(download_path)
                },
                'processed': file_info
            },
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Successfully processed California WARN data:")
        print(f"- {len(df)} records")
        print(f"- Download path: {download_path}")
        print(f"- CSV path: {file_info['csv']['path']}")
        print(f"- Parquet path: {file_info['parquet']['path']}")
        
    except Exception as e:
        print(f"Error processing California WARN data: {str(e)}")
        metadata['states_processed'].append({
            'state': 'california',
            'status': 'error',
            'error': str(e),
            'source_url': url
        })
    
    # Save metadata in the base directory
    metadata_path = os.path.join(base_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == '__main__':
    main()