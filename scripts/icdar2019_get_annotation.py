import xml.etree.ElementTree as ET
import os
import json
import pandas as pd

def polygon_to_bbox(points):
    coords = [tuple(map(int, point.split(','))) for point in points.split()]
    x_coords = [x for x, y in coords]
    y_coords = [y for x, y in coords]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    return [x1, y1, x2, y2]

def parse_table(table):
    cells = []
    for cell in table.findall('.//cell'):
        start_col = int(cell.get('start-col'))
        end_col = int(cell.get('end-col'))
        start_row = int(cell.get('start-row'))
        end_row = int(cell.get('end-row'))
        
        coords = cell.find('.//Coords').get('points')
        bbox = polygon_to_bbox(coords)

        cell_data = {
            "row_nums": list(range(start_row, end_row + 1)),
            "column_nums": list(range(start_col, end_col + 1)),
            "is_column_header": False,
            "is_projected_row_header": False,
            "box": bbox
        }
        cells.append(cell_data)
    return {"cells": cells}

def process_xml_file(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tables = root.findall('.//table')
        
        table_data = [parse_table(table) for table in tables]
        return table_data
    except ET.ParseError:
        print(f"Error parsing {xml_file}")
        return []

def parse_xml_to_dataframe(xml_file_path):
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError:
        return None
    
    width, height = int(root.get('right')), int(root.get('bottom'))
    data = []
    
    for block in root.findall('.//block'):
        block_id = block.get('id')
        for line in block.findall('.//line'):
            line_id = line.get('id')
            for word in line.findall('.//word'):
                x1 = int(word.get('left'))
                y1 = int(word.get('top'))
                x2 = int(word.get('right'))
                y2 = int(word.get('bottom'))

                text = word.get('value')
                confidence = int(word.get('confidence'))
                data.append((x1, y1, x2, y2, text, confidence, block_id, line_id))

    df = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2', 'text', 'confidence', 'block_id', 'line_id'])
    return df

def parse_json_to_dataframe(json_data):
    data = []

    for page in json_data['pages']:
        page_idx = page['page_idx']
        height, width = page['dimensions']

        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    x1, y1 = word['geometry'][0]
                    x2, y2 = word['geometry'][1]

                    x1 *= width
                    y1 *= height
                    x2 *= width
                    y2 *= height

                    text = word['value']
                    confidence = word['confidence']
                    block_id, line_id = -1, -1

                    data.append((x1, y1, x2, y2, text, confidence, page_idx, block_id, line_id))

    df = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2', 'text', 'confidence', 'page_idx', 'block_id', 'line_id'])
    return df

def calculate_iom(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    if bbox1_area == 0 or bbox2_area == 0:
        return 0.0

    iom = intersection_area / min(bbox1_area, bbox2_area)
    return iom

def assign_text_to_cells(ocr_df, annotation_json, iom_threshold=0.5):
    for table in annotation_json:
        for cell in table['cells']:
            cell['cell_text'] = ''
    
    for idx, word in ocr_df.iterrows():
        word_bbox = [word['x1'], word['y1'], word['x2'], word['y2']]
        best_iom = 0
        best_cell = None

        for table in annotation_json:
            for cell in table['cells']:
                cell_bbox = cell['box']
                iom = calculate_iom(word_bbox, cell_bbox)
                if iom > best_iom:
                    best_iom = iom
                    best_cell = cell

        if best_iom >= iom_threshold and best_cell is not None:
            if best_cell['cell_text'] == '':
                best_cell['cell_text'] = word['text']
            else:
                best_cell['cell_text'] += ' ' + word['text']

def process_files_in_directory(ocr_directory, annotation_directory, output_directory, substring, iom_threshold=0.5, ocr="ABBYY"):
    if ocr == "ABBYY":
        ocr_files = [f for f in os.listdir(ocr_directory) if substring in f and f.endswith('.xml')]
    elif ocr == "DOCTR":
        ocr_files = [f for f in os.listdir(ocr_directory) if substring in f and f.endswith('.json')]
    
    for ocr_file in ocr_files:
        ocr_file_path = os.path.join(ocr_directory, ocr_file)
        if ocr == "ABBYY":
            json_file_name = ocr_file.split('_')[0] + "_" + ocr_file.split('_')[1] + '.json'
            json_file_name = json_file_name.replace('ctdar', 'cTDaR')
            json_file_path = os.path.join(annotation_directory, json_file_name)
        elif ocr == "DOCTR":
            json_file_name = ocr_file
            json_file_path = os.path.join(annotation_directory, json_file_name)

        if not os.path.exists(json_file_path):
            print(f"Annotation JSON file not found for {ocr_file}")
            continue

        output_json_path = os.path.join(output_directory, json_file_name)

        if ocr == "ABBYY":
            ocr_df = parse_xml_to_dataframe(ocr_file_path)
        elif ocr == "DOCTR":
            with open(ocr_file_path, 'r') as ocr_file:
                ocr_json = json.load(ocr_file)
            ocr_df = parse_json_to_dataframe(ocr_json)
        else:
            print(f"OCR type {ocr} not recognized.")
            continue
        if ocr_df is None:
            print(f"Error parsing OCR XML file: {ocr_file_path}")
            continue

        with open(json_file_path, 'r') as json_file:
            annotation_json = json.load(json_file)
        
        assign_text_to_cells(ocr_df, annotation_json, iom_threshold)

        with open(output_json_path, 'w') as output_file:
            json.dump(annotation_json, output_file, indent=4)

        print(f"Processed {ocr_file_path} and saved results to {output_json_path}")

def convert_xml_to_json(input_directory, output_directory, substring):
    files = os.listdir(input_directory)
    matching_files = [file for file in files if substring in file]
    
    for file in matching_files:
        file_path = os.path.join(input_directory, file)
        table_data = process_xml_file(file_path)
        
        json_data = json.dumps(table_data, indent=4)
        json_file_path = os.path.join(output_directory, os.path.splitext(file)[0] + '.json')
        
        with open(json_file_path, 'w') as json_file:
            json_file.write(json_data)
        print(f"Processed {file} and saved to {json_file_path}")

# Example usage
ocr_directory = '/path/to/ocr_directory'
annotation_directory = '/path/to/annotation_directory'
output_directory = '/path/to/output_directory'
substring = '_t1'  # Change this substring as needed
ocr = "DOCTR"

process_files_in_directory(ocr_directory, annotation_directory, output_directory, substring, ocr=ocr)

input_directory = '/path/to/input_directory'
output_directory = '/path/to/output_directory'

convert_xml_to_json(input_directory, output_directory, substring)
