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
from typing import Tuple, Optional
from dataclasses import dataclass

from log_config import setup_logging

@dataclass
class AddressComponents:
    street_address: str
    unit: Optional[str]
    city: str
    state: str
    zipcode: str


# Set up logger at the module level
logger = setup_logging('warn_processor')



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


def preprocess_address(address: str) -> str:
    """
    Pre-process address string to handle edge cases before parsing.
    Specifically handles multiple units/suites separated by commas.
    """
    # Handle multiple units/suites separated by commas
    # Match patterns like: "Ste 100, 105, 106A, 106B" or "100, 200, 402"
    unit_list_pattern = r',\s*(?:\d+[A-Za-z]?(?:,\s*(?:\d+[A-Za-z]?))*)+(?=\s+[A-Za-z])'
    
    # Find all matches
    match = re.search(unit_list_pattern, address)
    if match:
        # Keep only the first unit number and remove the rest
        prefix = address[:match.start()]
        suffix = address[match.end():]
        first_unit = match.group(0).split(',')[0]
        address = f"{prefix}{first_unit}{suffix}"
    
    return address

def parse_address(address: str) -> AddressComponents:
    """
    Parse a full address string into components using usaddress library.
    Returns AddressComponents with street_address, unit, city, state, and zipcode.
    """
    import usaddress
    
    try:
        # Pre-process address to handle edge cases
        processed_address = preprocess_address(address)
        
        # Parse address using usaddress
        tagged_address, address_type = usaddress.tag(processed_address)
        
        # Extract components with fallbacks to empty string
        street_number = tagged_address.get('AddressNumber', '')
        pre_dir = tagged_address.get('StreetNamePreDirectional', '')
        street_name = tagged_address.get('StreetName', '')
        street_type = tagged_address.get('StreetNamePostType', '').rstrip(',')
        post_dir = tagged_address.get('StreetNamePostDirectional', '')
        
        # Combine street components
        street_parts = [p for p in [street_number, pre_dir, street_name, street_type, post_dir] if p]
        street_address = ' '.join(street_parts)
        
        # Get unit - combine various unit-related fields
        unit_parts = []
        if 'OccupancyType' in tagged_address:
            unit_parts.append(tagged_address['OccupancyType'])
        if 'OccupancyIdentifier' in tagged_address:
            unit_parts.append(tagged_address['OccupancyIdentifier'].rstrip(','))
        unit = ' '.join(unit_parts) if unit_parts else None
        
        # Get other components
        city = tagged_address.get('PlaceName', '')
        state = tagged_address.get('StateName', '')
        zipcode = tagged_address.get('ZipCode', '')
        
        # If parsing failed or components are missing, try regex fallback
        if not street_address or not city or not state:
            print(f"usaddress parsing incomplete for '{address}', trying fallback")
            return parse_address_fallback(address)
        
        return AddressComponents(
            street_address=street_address.strip(),
            unit=unit,
            city=city.strip(),
            state=state.strip(),
            zipcode=zipcode.strip()
        )
        
    except Exception as e:
        print(f"Error parsing address '{address}' with usaddress, trying fallback")
        return parse_address_fallback(address)

def parse_address_fallback(address: str) -> AddressComponents:
    """
    Fallback address parser using regex patterns when usaddress fails.
    """
    # Remove any extra whitespace and standardize separators
    address = re.sub(r'\s+', ' ', address).strip()
    
    # Try to extract zipcode
    zipcode_match = re.search(r'(\d{5})(?:-\d{4})?$', address)
    zipcode = zipcode_match.group(1) if zipcode_match else ''
    if zipcode:
        address = address[:zipcode_match.start()].strip()
    
    # Try to extract state
    state_pattern = (
        r'\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|'
        r'MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b'
    )
    state_match = re.search(state_pattern, address)
    state = state_match.group(1) if state_match else ''
    
    if state:
        # Split at state for city
        parts = address[:state_match.start()].strip().split(',')
        if len(parts) > 1:
            city = parts[-1].strip()
            street_address = ','.join(parts[:-1]).strip()
        else:
            # Try to split on last space before state
            pre_state = address[:state_match.start()].strip()
            last_space = pre_state.rfind(' ')
            if last_space > 0:
                street_address = pre_state[:last_space].strip()
                city = pre_state[last_space:].strip()
            else:
                street_address = pre_state
                city = ''
    else:
        street_address = address
        city = ''
    
    # Extract unit if present
    street_address, unit = extract_unit_from_street(street_address)
    
    return AddressComponents(
        street_address=street_address.strip(),
        unit=unit,
        city=city,
        state=state,
        zipcode=zipcode
    )

def extract_unit_from_street(street_address: str) -> Tuple[str, Optional[str]]:
    """
    Extract unit information from street address string.
    """
    unit_patterns = [
        # Match floor indicators (e.g., "4th Floor", "3rd Floor")
        r'\b(\d+(?:st|nd|rd|th)?\s*[Ff]loor)\b',
        # Match suite/unit numbers (e.g., "Suite 100", "Unit B", "#123")
        r'\b(?:Suite|Ste\.?|Unit|Apt\.?|#)\s*([A-Za-z0-9-]+)\b',
        # Match trailing unit numbers (e.g., "Building 3", "Room 4")
        r'\b(?:Building|Bldg\.?|Room|Rm\.?)\s*([A-Za-z0-9-]+)\b',
        # Match unit letters at end (e.g., "Unit A", "Suite B")
        r'\b([A-Za-z0-9-]+)\s*$'
    ]
    
    for pattern in unit_patterns:
        match = re.search(pattern, street_address, re.IGNORECASE)
        if match:
            unit = match.group(1)
            street_address = street_address[:match.start()].strip()
            return street_address, unit
            
    return street_address, None

def clean_street_address(street_address: str) -> str:
    """
    Clean and standardize street address component.
    """
    # Convert to uppercase for consistent matching
    address = street_address.upper()
    
    # Standardize directional abbreviations
    directions = {
        r'\bN\.?\s': 'NORTH ',
        r'\bS\.?\s': 'SOUTH ',
        r'\bE\.?\s': 'EAST ',
        r'\bW\.?\s': 'WEST ',
        r'\bNE\.?\s': 'NORTHEAST ',
        r'\bNW\.?\s': 'NORTHWEST ',
        r'\bSE\.?\s': 'SOUTHEAST ',
        r'\bSW\.?\s': 'SOUTHWEST '
    }
    for abbr, full in directions.items():
        address = re.sub(abbr, full, address)
    
    # Standardize street type abbreviations
    street_types = {
        r'\bAVE\.?\b': 'AVENUE',
        r'\bBLVD\.?\b': 'BOULEVARD',
        r'\bST\.?\b': 'STREET',
        r'\bRD\.?\b': 'ROAD',
        r'\bDR\.?\b': 'DRIVE',
        r'\bLN\.?\b': 'LANE',
        r'\bPKWY\.?\b': 'PARKWAY',
        r'\bPL\.?\b': 'PLACE',
        r'\bCT\.?\b': 'COURT',
        r'\bTER\.?\b': 'TERRACE',
        r'\bCIR\.?\b': 'CIRCLE'
    }
    for abbr, full in street_types.items():
        address = re.sub(abbr, full, address)
    
    # Remove multiple spaces
    address = re.sub(r'\s+', ' ', address)
    
    # Clean up any remaining special characters
    address = re.sub(r'[^\w\s.,-]', '', address)
    
    # Remove trailing punctuation except periods
    address = re.sub(r'[,-]+$', '', address)
    
    return address.strip()

def clean_address(address: str) -> str:
    """
    Clean and standardize a full address string.
    """
    # Parse address into components
    components = parse_address(address)
    
    # Clean the street address component
    cleaned_street = clean_street_address(components.street_address)
    
    # Reconstruct full address with cleaned components
    parts = [cleaned_street]
    if components.unit:
        parts.append(f"Unit {components.unit}")
    if components.city:
        parts.append(components.city)
    if components.state:
        parts.append(components.state)
    if components.zipcode:
        parts.append(components.zipcode)
    
    return " ".join(parts).strip()

def setup_geocoding_cache(base_dir: str) -> Tuple[pl.DataFrame, Path]:
    """
    Set up the geocoding cache with required schema.
    Creates new cache if existing cache doesn't match schema.
    """
    cache_dir = Path(base_dir) / 'geocoding' / 'ca'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'address_cache.parquet'
    
    # Define required schema
    required_schema = {
        'address': pl.String,
        'street_address': pl.String,
        'unit': pl.String,
        'city': pl.String,
        'state': pl.String,
        'zipcode': pl.String,
        'cleaned_street_address': pl.String,
        'latitude': pl.Float64,
        'longitude': pl.Float64,
        'last_updated': pl.String,
        'geocoding_source': pl.String
    }
    
    if cache_file.exists():
        try:
            existing_cache = pl.read_parquet(cache_file)
            current_schema = dict(existing_cache.schema)
            
            # Check if schemas match
            schema_matches = all(
                column in current_schema and current_schema[column] == dtype
                for column, dtype in required_schema.items()
            )
            
            if schema_matches:
                logger.info("Using existing cache with correct schema")
                return existing_cache, cache_file
            else:
                logger.warning("Cache schema mismatch. Creating new cache...")
                if cache_file.exists():
                    # Backup old cache with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_file = cache_file.with_suffix(f'.{timestamp}.bak')
                    cache_file.rename(backup_file)
                    logger.info(f"Backed up old cache to {backup_file}")
        except Exception as e:
            logger.error(f"Error reading cache file: {e}. Creating new cache.")
    
    # Create new cache with required schema
    cache_df = pl.DataFrame(schema=required_schema)
    cache_df.write_parquet(cache_file)
    logger.info("Created new cache with correct schema")
    
    return cache_df, cache_file

def add_to_cache(cache_df: pl.DataFrame, cache_file: Path, address: str, coordinates: Optional[Tuple[float, float]]) -> pl.DataFrame:
    """
    Add a new address and its coordinates to the cache.
    """
    logger.debug(f"\nAdding to cache:")
    logger.debug(f"Original address: {address}")
    
    # Parse address into components
    components = parse_address(address)
    logger.debug("\nParsed address components:")
    logger.debug(f"Street Address: {components.street_address}")
    logger.debug(f"Unit: {components.unit or 'N/A'}")
    logger.debug(f"City: {components.city}")
    logger.debug(f"State: {components.state}")
    logger.debug(f"Zip: {components.zipcode}")
    
    cleaned_street = clean_street_address(components.street_address)
    logger.debug(f"Cleaned street address: {cleaned_street}")
    
    # Create new row
    new_row = pl.DataFrame({
        'address': [address],
        'street_address': [components.street_address],
        'unit': [components.unit if components.unit else None],  # Ensure None for empty units
        'city': [components.city],
        'state': [components.state],
        'zipcode': [components.zipcode],
        'cleaned_street_address': [cleaned_street],
        'latitude': [coordinates[0] if coordinates else None],
        'longitude': [coordinates[1] if coordinates else None],
        'last_updated': [datetime.now().isoformat()],
        'geocoding_source': ['nominatim']
    })
    
    try:
        # Load current cache
        current_cache = pl.read_parquet(cache_file) if cache_file.exists() else cache_df
        logger.debug("Before update - cache entries: {}", current_cache.height)

        # Remove any existing entries for this address
        current_cache = current_cache.filter(
            ~(
                (pl.col('cleaned_street_address') == cleaned_street) &
                (pl.col('city') == components.city) &
                (pl.col('state') == components.state) &
                (pl.col('zipcode') == components.zipcode)
            )
        )
        logger.debug("After removing duplicates - cache entries: {}", current_cache.height)

        # Add new row and sort
        updated_cache = pl.concat([current_cache, new_row], how="vertical")
        updated_cache = updated_cache.sort('last_updated', descending=True)
        logger.debug("After adding new row - cache entries: {}", updated_cache.height)
        
        # Write to file
        updated_cache.write_parquet(cache_file)
        logger.debug("Cache successfully written to file")
        return updated_cache
        
    except Exception as e:
        print(f"Error updating cache: {e}")
        try:
            new_row.write_parquet(cache_file)
            logger.info("Wrote new row as new cache due to error")
        except Exception as write_error:
            logger.error(f"Error writing cache to disk: {write_error}")
        return new_row

def get_from_cache(cache_df: pl.DataFrame, address: str, cache_file: Path) -> Optional[Tuple[float, float]]:
    """
    Look up an address in the cache DataFrame.
    Returns (latitude, longitude) tuple if found, None if not found.
    """
    # Parse address into components for matching
    components = parse_address(address)
    cleaned_street = clean_street_address(components.street_address)
    
    try:
        # Attempt to read the latest cache from file
        current_cache = pl.read_parquet(cache_file) if cache_file.exists() else cache_df
        logger.debug("\nCache lookup debug:")
        logger.debug(f"Looking for address with components:")
        logger.debug(f"  cleaned_street_address: {cleaned_street}")
        logger.debug(f"  city: {components.city}")
        logger.debug(f"  state: {components.state}")
        logger.debug(f"  zipcode: {components.zipcode}")
        
        # Match on cleaned street address and other components
        result = current_cache.filter(
            (pl.col('cleaned_street_address') == cleaned_street) &
            (pl.col('city') == components.city) &
            (pl.col('state') == components.state) &
            (pl.col('zipcode') == components.zipcode)
        )
        
        if result.height > 0:
            logger.debug(f"Found {result.height} matching entries in cache")
            # Get the most recent entry (should be first due to sorting)
            row = result.row(0)
            lat, lon = row[7], row[8]  # indices for latitude and longitude
            logger.debug(f"Coordinates from cache: lat={lat}, lon={lon}")
            if lat is not None and lon is not None:
                return (lat, lon)
            logger.debug("Cached entry has no coordinates")
        else:
            logger.debug("No matching entries found in cache")
            # Debug: Show what's in the cache
            print("\nFirst few cache entries:")
            if current_cache.height > 0:
                logger.debug(current_cache.select([
                    'cleaned_street_address', 'city', 'state', 'zipcode'
                ]).head(3))
            else:
                logger.debug("Cache is empty")
    except Exception as e:
        logger.error(f"Error reading cache file: {e}, falling back to memory cache")
        current_cache = cache_df
    
    return None

def geocode_address(address_string: str, cache_df: pl.DataFrame, cache_file: Path) -> Optional[Tuple[float, float]]:
    """
    Geocode a full address string to obtain latitude and longitude.
    Uses cache stored in polars DataFrame.
    """
    try:
        # Check cache first
        cached_result = get_from_cache(cache_df, address_string, cache_file)
        if cached_result is not None:
            if any(coord is None for coord in cached_result):
                # Check if we should retry missing coordinates
                if os.getenv('RETRY_MISSING_COORDINATES', '').lower() in ('true', '1', 'yes'):
                    logger.debug(f"Retrying previously failed address: {address_string}")
                else:
                    logger.info(f"Skipping previously failed address: {address_string}")
                    return None
            else:
                logger.debug(f"Cache hit for address: {address_string}")
                return cached_result
            
        logger.debug(f"Cache miss for address: {address_string}")
        
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
        components = parse_address(address_string)
        clean_addr = f"{components.street_address}, {components.city}, {components.state} {components.zipcode}"
        location = geocode(clean_addr)
        result = (location.latitude, location.longitude) if location else None
            
        # Always add to cache, even if geocoding failed
        cache_df = add_to_cache(cache_df, cache_file, address_string, result)
        return result
        
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.error(f"Geocoding error for address '{address_string}': {str(e)}")
        # Add failed attempt to cache
        cache_df = add_to_cache(cache_df, cache_file, address_string, None)
        return None
    except Exception as e:
        logger.error(f"Unexpected error geocoding address '{address_string}': {str(e)}")
        # Add failed attempt to cache
        cache_df = add_to_cache(cache_df, cache_file, address_string, None)
        return None

def process_california_warn(file_path: str, base_dir: str) -> pl.DataFrame:
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
    logger.info(f"Geocoding {len(addresses)} addresses...")
    
    # Process addresses with progress indicator
    coordinates = []
    retry_flag = os.getenv('RETRY_MISSING_COORDINATES', '').lower() in ('true', '1', 'yes')
    
    for i, addr in enumerate(addresses, 1):
        logger.info(f"\nProcessing address {i}/{len(addresses)}: {addr}")
        coord = geocode_address(addr, cache_df, cache_file)
        coordinates.append(coord)
    
    # Add latitude and longitude columns
    df = df.with_columns([
        pl.Series('latitude', [coord[0] if coord else None for coord in coordinates]),
        pl.Series('longitude', [coord[1] if coord else None for coord in coordinates])
    ])
    
    # Calculate geocoding success rate
    total_addresses = len(addresses)
    successful_geocodes = sum(1 for coord in coordinates if coord is not None)
    success_rate = (successful_geocodes / total_addresses) * 100 if total_addresses > 0 else 0
    
    logger.info(f"\nGeocoding Summary:")
    logger.info(f"Total addresses: {total_addresses}")
    logger.info(f"Successfully geocoded: {successful_geocodes}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"Retry missing coordinates flag: {retry_flag}")
    
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
    df.write_csv(csv_path)
    file_info['csv'] = {
        'path': csv_path,
        'size_bytes': os.path.getsize(csv_path)
    }

    # Save as Parquet
    parquet_path = os.path.join(processed_dir, f'{base_name}.parquet')
    df.write_parquet(parquet_path)
    file_info['parquet'] = {
        'path': parquet_path,
        'size_bytes': os.path.getsize(parquet_path)
    }

    return file_info

def main():
    # Get base directory from command line argument
    if len(sys.argv) != 2:
        logger.error("Usage: python download_warn.py BASE_DIR")
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
        logger.info("Downloading California WARN data...")
        url = get_california_warn_url()
        download_path = download_file(url, dirs['download_dir'])
        
        # Process the file
        logger.info("Processing California WARN data...")
        df = process_california_warn(download_path, base_dir)
        
        # Save processed files
        logger.info("Saving processed data...")
        file_info = save_processed_files(df, dirs['processed_dir'], dirs['year'])
        
        # Log success metrics
        logger.info(f"Successfully processed {df.height} records")
        logger.debug(f"Files saved: {file_info}")
        
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
        logger.error(f"Error processing California WARN data: {str(e)}")
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