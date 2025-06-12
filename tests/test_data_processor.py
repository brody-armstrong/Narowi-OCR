import pytest
from datetime import datetime
from pathlib import Path
import json
import tempfile
import shutil

from src.processing.data_processor import (
    DataProcessor, ProcessingStatus, ProcessingResult, BatchResult
)
from src.processing.pattern_matcher import MedicalReading, ReadingType

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_readings():
    """Create sample medical readings for testing."""
    return [
        MedicalReading(
            type=ReadingType.TEMPERATURE,
            value=98.6,
            unit='F',
            is_valid=True
        ),
        MedicalReading(
            type=ReadingType.OXYGEN,
            value=98,
            unit='%',
            is_valid=True
        ),
        MedicalReading(
            type=ReadingType.HEART_RATE,
            value=72,
            unit='BPM',
            is_valid=False  # Invalid reading for testing
        )
    ]

@pytest.fixture
def sample_image_results(sample_readings):
    """Create sample image results for batch processing."""
    return [
        {
            'image_path': 'test_image1.jpg',
            'readings': sample_readings,
            'confidence': 95.0,
            'metadata': {'device_type': 'oximeter'}
        },
        {
            'image_path': 'test_image2.jpg',
            'readings': [sample_readings[0]],  # Only temperature
            'confidence': 98.0,
            'metadata': {'device_type': 'thermometer'}
        }
    ]

def test_process_single_image(temp_dir, sample_readings):
    """Test processing a single image."""
    processor = DataProcessor(output_dir=temp_dir)
    
    result = processor.process_single_image(
        image_path='test_image.jpg',
        readings=sample_readings,
        confidence=95.0,
        metadata={'device_type': 'oximeter'}
    )
    
    assert result.image_path == 'test_image.jpg'
    assert result.status == ProcessingStatus.PARTIAL  # Because one reading is invalid
    assert len(result.readings) == 3
    assert result.confidence == 95.0
    assert result.error_message is not None
    assert result.metadata['device_type'] == 'oximeter'
    
    # Check if result was saved
    result_files = list(Path(temp_dir).glob('*.json'))
    assert len(result_files) == 1

def test_process_batch(temp_dir, sample_image_results):
    """Test batch processing of multiple images."""
    processor = DataProcessor(output_dir=temp_dir)
    
    batch_result = processor.process_batch(sample_image_results)
    
    assert batch_result.total_images == 2
    assert batch_result.successful_images == 1  # One image has all valid readings
    assert batch_result.failed_images == 0  # No completely failed images
    assert len(batch_result.results) == 2
    
    # Check summary statistics
    assert batch_result.summary['total_readings'] == 4  # 3 + 1 readings
    assert batch_result.summary['valid_readings'] == 3  # 2 valid + 1 valid
    assert batch_result.summary['average_confidence'] == 96.5  # (95.0 + 98.0) / 2
    
    # Check reading type counts
    type_counts = batch_result.summary['reading_types']
    assert type_counts[ReadingType.TEMPERATURE.name] == 2  # Two temperature readings
    assert type_counts[ReadingType.OXYGEN.name] == 1  # One oxygen reading
    assert type_counts[ReadingType.HEART_RATE.name] == 0  # One heart rate reading but invalid
    
    # Check if batch result was saved
    result_files = list(Path(temp_dir).glob('batch_*.json'))
    assert len(result_files) == 1

def test_export_to_csv(temp_dir, sample_image_results):
    """Test exporting batch results to CSV."""
    processor = DataProcessor(output_dir=temp_dir)
    batch_result = processor.process_batch(sample_image_results)
    
    csv_path = processor.export_to_csv(batch_result)
    
    # Check if CSV file was created
    assert Path(csv_path).exists()
    
    # Read CSV and verify content
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        
    # Check header
    assert 'Image Path,Timestamp,Status,Reading Type,Value,Unit,Is Valid,Confidence' in lines[0]
    
    # Check data rows (should have 4 rows - 3 readings from first image + 1 from second)
    assert len(lines) == 5  # Header + 4 data rows

def test_error_handling(temp_dir):
    """Test error handling in data processing."""
    processor = DataProcessor(output_dir=temp_dir)
    
    # Test with invalid data
    result = processor.process_single_image(
        image_path='test_image.jpg',
        readings=[],  # Empty readings
        confidence=0.0
    )
    
    assert result.status == ProcessingStatus.FAILED
    assert result.error_message == "No readings found"
    assert len(result.readings) == 0
    
    # Test with invalid batch data
    batch_result = processor.process_batch([
        {
            'image_path': 'test_image.jpg',
            'readings': [],
            'confidence': 0.0
        }
    ])
    
    assert batch_result.total_images == 1
    assert batch_result.successful_images == 0
    assert batch_result.failed_images == 1
    assert batch_result.summary['total_readings'] == 0
    assert batch_result.summary['valid_readings'] == 0

def test_metadata_handling(temp_dir, sample_readings):
    """Test handling of metadata in processing results."""
    processor = DataProcessor(output_dir=temp_dir)
    
    metadata = {
        'device_type': 'oximeter',
        'device_id': 'OX123',
        'patient_id': 'P456',
        'timestamp': datetime.now().isoformat()
    }
    
    result = processor.process_single_image(
        image_path='test_image.jpg',
        readings=sample_readings,
        confidence=95.0,
        metadata=metadata
    )
    
    assert result.metadata == metadata
    
    # Verify metadata is saved in JSON
    result_files = list(Path(temp_dir).glob('*.json'))
    with open(result_files[0], 'r') as f:
        saved_data = json.load(f)
        assert saved_data['metadata'] == metadata 