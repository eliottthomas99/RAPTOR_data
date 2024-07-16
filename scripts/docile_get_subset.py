import os
import json
import numpy as np
import pandas as pd

def centroid(bbox):
    """Calculate centroid of a bounding box."""
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2, (y0 + y1) / 2

def bbox_similarity(bbox1, bbox2, threshold=0.01):
    """Check if two bounding boxes are similar."""
    return np.linalg.norm(np.array(centroid(bbox1)) - np.array(centroid(bbox2))) < threshold

def parse_json_to_dataframe(json_data, page_target=None):
    data = []

    for page in json_data['pages']:
        page_idx = page['page_idx']
        if page_target == page_idx:
            width, height = page['dimensions']

            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        x1, y1 = word['geometry'][0]
                        x2, y2 = word['geometry'][1]

                        text = word['value']
                        confidence = word['confidence']
                        block_id, line_id = -1, -1

                        data.append((x1, y1, x2, y2, text, confidence, page_idx, block_id, line_id))

    df = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2', 'text', 'confidence', 'page_idx', 'block_id', 'line_id'])
    return df

def intersection_over_minimum(bbox1, bbox2):
    """Calculate the Intersection over Minimum (IoM) of two bounding boxes."""
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

def iom_x1(bbox1, bbox2):
    """Calculate IoM of the x (x2-x1) of two bboxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    intersection = min(x2_1, x2_2) - max(x1_1, x1_2)

    if intersection < 0:
        return 0.0
    
    return intersection / min(x2_1 - x1_1, x2_2 - x1_2)

def get_row_cells(annotation, ocr_df, page_index, extractions):
    cells = []
    page_data = annotation['metadata']['page_to_table_grid'][str(page_index)]
    rows = page_data['rows']
    columns = page_data['columns']

    for extraction in extractions:
        extraction_bbox = extraction['bbox']
        ocr_df = ocr_df[
            ocr_df.apply(lambda word: not intersection_over_minimum(
                [word['x1'], word['y1'], word['x2'], word['y2']], extraction_bbox) > 0.8, axis=1)
        ]

    data_rows = 0
    for row in rows:
        if row['row_type'] == 'data':
            data_rows += 1
        if row['row_type'] in ['up-merge', 'down-merge']:
            row_bbox = [
                page_data['bbox'][0],
                row['top'],
                page_data['bbox'][2],
                row['bottom']
            ]

            words_in_row = ocr_df[
                ocr_df.apply(lambda word: intersection_over_minimum(
                    [word['x1'], word['y1'], word['x2'], word['y2']], row_bbox) > 0.9, axis=1)
            ]

            for col in columns:
                col_bbox = [
                    col['left'],
                    row['top'],
                    col['right'],
                    row['bottom']
                ]
                words_in_col = words_in_row[
                    words_in_row.apply(lambda word: intersection_over_minimum(
                        [word['x1'], word['y1'], word['x2'], word['y2']], col_bbox) > 0.9, axis=1)
                ]

                if not words_in_col.empty:
                    merged_text = ' '.join(words_in_col['text'].tolist())
                    merged_bbox = [
                        words_in_col['x1'].min(),
                        words_in_col['y1'].min(),
                        words_in_col['x2'].max(),
                        words_in_col['y2'].max()
                    ]

                    cell = {
                        "bbox": merged_bbox,
                        "column_nums": [columns.index(col)],
                        "row_nums": [data_rows], 
                        "column header": False,
                        "subcell": False,
                        "projected row header": False,
                        "cell_text": merged_text,
                        "full_bbox": [],
                        "img_size": [],
                        "row_type": row['row_type']
                    }
                    cells.append(cell)
    return cells

def extract_cells(annotation, ocr_df, page_index, filename):
    cells = []
    rows = annotation['metadata']['page_to_table_grid'][str(page_index)]['rows']
    extractions = annotation.get('line_item_extractions', [])
    present = False
    for row in rows:
        if row['row_type'] in ['up-merge', 'down-merge']:
            present = True
            break
            
    if present:
        row_cells = get_row_cells(annotation, ocr_df, page_index, extractions)
    else:
        row_cells = []
    
    headers = annotation.get('line_item_headers', [])
    headers = [header for header in headers if header['page'] == page_index]

    if any(header['line_item_id'] > 0 for header in headers):
        return []

    for extraction in extractions:
        if extraction['bbox'][3] < headers[0]['bbox'][1]:
            return []

    unique_headers = {}
    for header in headers:
        key = (header['fieldtype'], header['text'])
        if key not in unique_headers:
            unique_headers[key] = header
        else:
            if not bbox_similarity(header['bbox'], unique_headers[key]['bbox']):
                unique_headers[key] = header

    header_positions = {}
    unique_headers = dict(sorted(unique_headers.items(), key=lambda x: x[1]['bbox'][0]))
    for i, (key, header) in enumerate(unique_headers.items()):
        header_positions[header['fieldtype']] = header
        header_text = header.get('text', '').replace('\n', ' ')
        cell = {
            "bbox": header['bbox'],
            "column_nums": [i],
            "row_nums": [0],
            "column header": True,
            "subcell": False,
            "projected row header": False,
            "cell_text": header_text,
            "full_bbox": [],
            "img_size": []
        }
        cells.append(cell)

    for extraction in extractions:
        extraction_bbox = extraction['bbox']
        extraction_centroid = centroid(extraction_bbox)

        closest_header = None
        min_distance = float('inf')
        same_class_headers = [header for header in unique_headers.values() if header['fieldtype'] == extraction['fieldtype']]

        if same_class_headers:
            for header in same_class_headers:
                header_centroid = centroid(header['bbox'])
                distance = np.linalg.norm(np.array(extraction_centroid) - np.array(header_centroid))

                if distance < min_distance:
                    min_distance = distance
                    closest_header = header

        if closest_header is None:
            continue

        column_num = list(unique_headers.values()).index(closest_header)
        if column_num == -1:
            continue

        row_num = extraction['line_item_id']

        text = extraction.get('text', '').replace('\n', ' ')
        cell = {
            "bbox": extraction['bbox'],
            "column_nums": [column_num],
            "row_nums": [row_num],
            "column header": False,
            "subcell": False,
            "projected row header": False,
            "cell_text": text,
            "full_bbox": [],
            "img_size": []
        }

        merged = False
        for row_cell in row_cells:
            if row_cell['column_nums'][0] != column_num:
                continue
            row_num_diff = row_cell['row_nums'][0] - row_num
            if row_num_diff == -1 and row_cell['row_type'] == 'down-merge':
                merged = True
                cell['cell_text'] = row_cell['cell_text'] + ' ' + cell['cell_text']
                cell['bbox'] = [
                    min(cell['bbox'][0], row_cell['bbox'][0]),
                    row_cell['bbox'][1],
                    max(cell['bbox'][2], row_cell['bbox'][2]),
                    cell['bbox'][3]
                ]
            if row_num_diff == 1 and row_cell['row_type'] == 'up-merge':
                merged = True
                cell['cell_text'] = cell['cell_text'] + ' ' + row_cell['cell_text']
                cell['bbox'] = [
                    min(cell['bbox'][0], row_cell['bbox'][0]),
                    cell['bbox'][1],
                    max(cell['bbox'][2], row_cell['bbox'][2]),
                    row_cell['bbox'][3]
                ]

            if merged:
                row_cells.remove(row_cell)
                break
        cells.append(cell)
    return cells

def read_annotation(ocr_json_path):
    with open(ocr_json_path) as f:
        return json.load(f)

def process_file(ocr_json_path, output_path):
    annotation = read_annotation(ocr_json_path)
    ocr_df = parse_json_to_dataframe(annotation)

    cells = []
    for page_index in range(1, 2):
        cells.extend(extract_cells(annotation, ocr_df, page_index, ocr_json_path))

    with open(output_path, 'w') as f:
        json.dump(cells, f)

if __name__ == "__main__":
    ocr_json_path = 'path_to_your_ocr_json_file.json'
    output_path = 'path_to_save_output.json'
    process_file(ocr_json_path, output_path)
