import os
import re
import glob

# Speaker mapping data from your table
speaker_mapping = {
    '5': {
        'SPEAKER_02': 'DARRYL',
        'SPEAKER_01': 'PAMELA', 
        'SPEAKER_00': 'PAMELA',
        'None': 'DARRYL'
    },
    '6': {
        'SPEAKER_01': 'ALINA',
        'SPEAKER_02': 'LENORE',
        'SPEAKER_00': 'ALINA'
    },
    '7': {
        'SPEAKER_00': 'ALICE',
        'SPEAKER_01': 'MARY',
        'SPEAKER_02': 'MARY'
    },
    '9': {
        'SPEAKER_00': 'NATHAN',
        'SPEAKER_01': 'KATHY',
        'SPEAKER_02': 'KATHY'
    },
    '17': {
        'SPEAKER_00': 'MICHAEL',
        'SPEAKER_01': 'JIM',
        'SPEAKER_02': 'MICHAEL'
    },
    '22': {
        'SPEAKER_00': 'LANCE',
        'SPEAKER_01': 'RANDY',
        'SPEAKER_02': 'LANCE'
    },
    '24': {
        'SPEAKER_01': 'JENNIFER',
        'SPEAKER_02': 'DAN',
        'SPEAKER_00': 'JENNIFER'
    },
    '28': {
        'SPEAKER_00': 'JEFF',
        'SPEAKER_01': 'JILL',
        'SPEAKER_02': 'JEFF'
    },
    '29': {
        'SPEAKER_00': 'LARRY',
        'SPEAKER_01': 'LARRY',
        'SPEAKER_02': 'SETH'
    },
    '34': {
        'SPEAKER_00': 'SCOTT',
        'SPEAKER_01': 'KAREN',
        'SPEAKER_02': 'SCOTT',
        'None': 'KAREN'
    },
    '41': {
        'SPEAKER_00': 'KRISTIN',
        'SPEAKER_01': 'PAIGE'
    },
    '43': {
        'SPEAKER_00': 'ALICE',
        'SPEAKER_01': 'ANNETTE',
        'SPEAKER_02': 'ALICE',
        'None': 'ALICE'
    },
    '44': {
        'SPEAKER_00': 'CAM',
        'SPEAKER_01': 'LAJUAN'
    },
    '45': {
        'SPEAKER_00': 'CORINNA',
        'SPEAKER_01': 'PATRICK',
        'SPEAKER_02': 'PATRICK'
    },
    '47': {
        'SPEAKER_00': 'RICHARD',
        'SPEAKER_01': 'RICHARD',
        'SPEAKER_02': 'FRED',
        'None': 'RICHARD'
    },
    '58': {
        'SPEAKER_00': 'SHERI',
        'SPEAKER_01': 'STEVEN',
        'None': 'SHERI'
    },
    '60': {
        'SPEAKER_01': 'JON',
        'SPEAKER_00': 'ALAN'
    }
}

def replace_speakers(input_path, output_path):
    """Process a single transcript file and replace speaker labels"""
    
    # Extract the number from filename (e.g., "0005" from "SBC0005_diarized_transcript.txt")
    filename = os.path.basename(input_path)
    match = re.search(r'SBC(\d+)_diarized_transcript\.txt', filename)
    
    if not match:
        print(f"Could not extract number from filename: {filename}")
        return
    
    file_number = match.group(1)
    # Remove leading zeros and get the key
    key = str(int(file_number))
    
    if key not in speaker_mapping:
        print(f"No speaker mapping found for file number: {key}")
        return
    
    mapping = speaker_mapping[key]
    
    # Read the file
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace all speaker labels
    for speaker_label, speaker_name in mapping.items():
        # Handle different spacing variations
        patterns = [
            f"{speaker_label} ",
            f"{speaker_label}:",
            f"{speaker_label}: ",
            f" {speaker_label}"
        ]
        
        for pattern in patterns:
            if speaker_label == 'NONE':
                # Special handling for NONE
                content = re.sub(r'\bNONE:\s*', f'{speaker_name}: ', content)
                content = re.sub(r'\bNONE\s+', f'{speaker_name}: ', content)
            else:
                # Replace with colon format for consistency
                content = content.replace(pattern, f'{speaker_name}: ')
    
    # Clean up any double spaces created by replacements
    content = re.sub(r' +', ' ', content)
    # Clean up any double colons
    content = re.sub(r':+', ':', content)
    
    # Write the modified content back
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Processed: {filename} -> {os.path.basename(output_path)}")

def process_transcript_file(file_path):
    """Alternative function that uses the same logic but returns the output path"""
    filename = os.path.basename(file_path)
    output_path = file_path.replace('_diarized_transcript.txt', '_named_transcript.txt')
    replace_speakers(file_path, output_path)
    return output_path

# Main execution
input_directory = "/Users/nsingh8/Documents/projects/DiarizationLM/input_transcripts"
output_directory = "/Users/nsingh8/Documents/projects/DiarizationLM/outout_dlm"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process files 5 to 60
for i in range(5, 61):
    input_filename = f"SBC{str(i).zfill(4)}_diarized_transcript.txt"
    output_filename = f"SBC{str(i).zfill(4)}_named_transcript.txt"
    
    input_path = os.path.join(input_directory, input_filename)
    output_path = os.path.join(output_directory, output_filename)
    
    if os.path.exists(input_path):
        replace_speakers(input_path, output_path)
    else:
        print(f"File not found: {input_path}")

print("Processing complete!")