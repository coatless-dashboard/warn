import requests
import polars as pl
import os
from datetime import datetime
import json
import sys
import re
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.extra.rate_limiter import RateLimiter
import time

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

def standardize_column_name(col_name):
    """
    Standardize column names:
    1. Convert to lowercase
    2. Remove special characters
    3. Replace whitespace with underscore
    4. Remove any remaining invalid characters
    """
    # Convert to string and lowercase
    col_name = str(col_name).lower()
    
    # Replace common special characters and whitespace with underscore
    col_name = re.sub(r'[-./\s]+', '_', col_name)
    
    # Remove any other special characters
    col_name = re.sub(r'[^a-z0-9_]', '', col_name)
    
    # Remove leading/trailing underscores
    col_name = col_name.strip('_')
    
    # Replace multiple underscores with single underscore
    col_name = re.sub(r'_+', '_', col_name)
    
    return col_name


def clean_address(address):
    """
    Clean and standardize address string:
    - Remove suite/unit numbers
    - Remove extra whitespace
    - Handle special characters
    """
    # Convert to string if not already
    address = str(address)
    
    # Remove suite/unit numbers (e.g., "Suite #2300", "# 1100")
    address = re.sub(r'(?i)(?:suite|ste\.?|unit|#)\s*#?\s*\d+,?\s*', '', address)
    
    # Remove multiple spaces
    address = re.sub(r'\s+', ' ', address)
    
    # Clean up any remaining special characters
    address = re.sub(r'[^\w\s,.-]', '', address)
    
    return address.strip()

def setup_geocoding_cache(base_dir):
    """
    Set up the geocoding cache using a polars DataFrame.
    Creates and loads cache if it exists, otherwise creates new cache.
    """
    cache_dir = Path(base_dir) / 'geocoding' / 'ca'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'address_cache.parquet'
    
    if cache_file.exists():
        try:
            # Load existing cache
            cache_df = pl.read_parquet(cache_file)
            # Validate schema
            required_columns = {
                'address': pl.String,
                'cleaned_address': pl.String,
                'latitude': pl.Float64,
                'longitude': pl.Float64,
                'last_updated': pl.String,
                'geocoding_source': pl.String
            }
            
            # Check if all required columns exist with correct types
            current_schema = dict(cache_df.schema)
            if not all(col in current_schema and current_schema[col] == dtype 
                      for col, dtype in required_columns.items()):
                print("Cache file schema mismatch, creating new cache")
                cache_df = pl.DataFrame(schema=required_columns)
        except Exception as e:
            print(f"Error reading cache file: {e}. Creating new cache.")
            cache_df = pl.DataFrame(schema=required_columns)
    else:
        # Create new cache with explicit schema
        cache_df = pl.DataFrame(schema={
            'address': pl.String,
            'cleaned_address': pl.String,
            'latitude': pl.Float64,
            'longitude': pl.Float64,
            'last_updated': pl.String,
            'geocoding_source': pl.String
        })
    
    return cache_df, cache_file

def add_to_cache(cache_df, cache_file, address, coordinates):
    """
    Add a new address and its coordinates to the cache.
    Updates cache file on disk after adding.
    """
    clean_addr = clean_address(address)
    
    # Create new row
    new_row = pl.DataFrame({
        'address': [address],
        'cleaned_address': [clean_addr],
        'latitude': [coordinates[0] if coordinates else None],
        'longitude': [coordinates[1] if coordinates else None],
        'last_updated': [datetime.now().isoformat()],
        'geocoding_source': ['nominatim']
    })
    
    try:
        # Load the most recent cache from file
        current_cache = pl.read_parquet(cache_file) if cache_file.exists() else cache_df
        
        # Remove any existing entries for this address
        current_cache = current_cache.filter(pl.col('cleaned_address') != clean_addr)
        
        # Concatenate the new row with the existing cache
        updated_cache = pl.concat([current_cache, new_row], how="vertical")
        
        # Sort by last_updated to keep most recent entries first
        updated_cache = updated_cache.sort('last_updated', descending=True)
        
        # Write the entire updated cache back to file
        updated_cache.write_parquet(cache_file)
        
        return updated_cache
        
    except Exception as e:
        print(f"Error updating cache: {e}")
        # If there's an error, try to at least save the current cache_df
        try:
            cache_df.write_parquet(cache_file)
        except Exception as write_error:
            print(f"Error writing cache to disk: {write_error}")
        return cache_df

def get_from_cache(cache_df, address, cache_file):
    """
    Look up an address in the cache DataFrame.
    Returns (latitude, longitude) tuple if found, None if not found.
    """
    clean_addr = clean_address(address)
    
    try:
        # Attempt to read the latest cache from file
        current_cache = pl.read_parquet(cache_file) if cache_file.exists() else cache_df
    except Exception as e:
        print(f"Error reading cache file: {e}, falling back to memory cache")
        current_cache = cache_df
    
    result = current_cache.filter(pl.col('cleaned_address') == clean_addr)
    
    if result.height > 0:
        # Get the most recent entry (should be first due to sorting)
        row = result.row(0)
        if row[2] is not None and row[3] is not None:  # Check lat/lon not null
            return (row[2], row[3])
    return None

def geocode_address(address_string, cache_df, cache_file):
    """
    Geocode a full address string to obtain latitude and longitude.
    Uses cache stored in polars DataFrame.
    """
    try:
        # Check cache first
        cached_result = get_from_cache(cache_df, address_string, cache_file)
        if cached_result is not None:
            print(f"Cache hit for address: {address_string}")
            return cached_result
            
        print(f"Cache miss for address: {address_string}")
        
        # Initialize the geocoder with increased timeout
        geolocator = Nominatim(
            user_agent="warn_notice_processor",
            timeout=5
        )
        
        # Create a rate-limited version of the geocoding function
        geocode = RateLimiter(
            geolocator.geocode,
            min_delay_seconds=1,
            max_retries=2,
            error_wait_seconds=2.0
        )
        
        # Try geocoding with the cleaned address
        location = geocode(clean_address(address_string))
        if location:
            result = (location.latitude, location.longitude)
            # Update cache with new result
            cache_df = add_to_cache(cache_df, cache_file, address_string, result)
            return result
            
        # Store failed attempt in cache
        cache_df = add_to_cache(cache_df, cache_file, address_string, None)
        return None
        
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error for address '{address_string}': {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error geocoding address '{address_string}': {str(e)}")
        return None

def process_california_warn(file_path, base_dir):
    """
    Process California WARN Excel file with geocoding cache support.
    """
    # Set up cache
    cache_df, cache_file = setup_geocoding_cache(base_dir)
    
    # Read Excel file using polars
    df = pl.read_excel(
        source=file_path,
        sheet_name='Detailed WARN Report ',
        engine='calamine',
        read_options={
            'header_row': 1,
            'skip_rows': 0
        }
    )
    
    # Standardize column names
    df = df.select([
        pl.col(col).alias(standardize_column_name(col))
        for col in df.columns
    ])

    # Add state and download timestamp
    df = df.with_columns([
        pl.lit('california').alias('state'),
        pl.lit(datetime.now().isoformat()).alias('download_date')
    ])
    
    # Geocode addresses
    addresses = df.select(['address']).to_series().to_list()
    print(f"Geocoding {len(addresses)} addresses...")
    
    # Process addresses with progress indicator
    coordinates = []
    for i, addr in enumerate(addresses, 1):
        print(f"Processing address {i}/{len(addresses)}: {addr}")
        coord = geocode_address(addr, cache_df, cache_file)
        coordinates.append(coord)
    
    # Add latitude and longitude columns
    df = df.with_columns([
        pl.Series('latitude', [coord[0] if coord else None for coord in coordinates]),
        pl.Series('longitude', [coord[1] if coord else None for coord in coordinates])
    ])
    
    return df

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
        df = process_california_warn(download_path, base_dir)
        
        # Save processed files
        print("Saving processed data...")
        file_info = save_processed_files(df, dirs['processed_dir'], dirs['year'])
        
        metadata['states_processed'].append({
            'state': 'california',
            'status': 'success',
            'records': df.height,
            'source_url': url,
            'columns': df.columns,
            'files': {
                'download': {
                    'path': str(download_path),
                    'size_bytes': os.path.getsize(download_path)
                },
                'processed': file_info
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error processing California WARN data: {str(e)}")
        metadata['states_processed'].append({
            'state': 'california',
            'status': 'error',
            'error': str(e),
            'source_url': url
        })
    
    # Save metadata
    metadata_path = os.path.join(base_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == '__main__':
    main()