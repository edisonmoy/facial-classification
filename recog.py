import json
import os
from watson_developer_cloud import VisualRecognitionV3

path = 'downloads'

visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_apikey='2Jqso7bRXMN8VYzkBYLRnPP0pYc-6dsD84QryGNZfvn7')

for directories in os.listdir(path):
    for folder in directories:
        filename = os.path.join(folder, directories)
        with open(filename, 'rb') as images_file:
            faces = visual_recognition.detect_faces(images_file).get_result()
            print(json.dumps(faces, indent=2))
