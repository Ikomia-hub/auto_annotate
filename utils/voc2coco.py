# SOURCE: This script is an adaptation of the script from the following link: 
# https://www.aitude.com/annotation-converters-for-object-detection/


import xml.etree.ElementTree as ET
import json
import os
import json
import xml.etree.ElementTree as ET
import glob

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None


# Write the list of XML file paths to a text file
def get_xml_files(folder_path, ratio):
    xml_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))

    split_index = int(len(xml_files) * ratio)
    xml_files_train = xml_files[:split_index]
    xml_files_test = xml_files[split_index:]
    
    return xml_files_train, xml_files_test

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars

def get_filename(filename):
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return str(filename)
    
def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.
        Arguments:
        xml_files {list} -- A list of xml file paths.
    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}

def convert(xml_files, json_file, task):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = get_filename(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)

        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)

            polygons = []
            if task == "segmentation":
            # Get the polygon vertices
                polygon = get_and_check(obj, "polygon", 1)
                len_poly = int(len(polygon) / 2)
                # for element in polygon:
                for i in range(1, len_poly):
                    polygons.append(int(get_and_check(polygon, "x" + str(i), 1).text))
                    polygons.append(int(get_and_check(polygon, "y" + str(i), 1).text))
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [polygons],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    categories_list = []
    for name, id in categories.items():
        item = {"id": id, "name": name}
        categories_list.append(item)

    for cat in categories_list:
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
