# importing google_images_download module
from google_images_download import google_images_download

# creating object
response = google_images_download.googleimagesdownload()

# search_queries = ['chinese_man_face','chinese_woman_face', 'chinese_male_face','chinese_female_face','chinese_adult_face','chinese_person_face', 
# 'ghanaian_man_face','ghanaian_woman_face','ghanaian_male_face','ghanaian_female_face','ghanaian_adult_face','ghanaian_person_face']

search_queries = ['chinese_child_face', 'ghanaian_child_face']


def downloadimages(query):
    # keywords is the search query
    # format is the image file format
    # limit is the number of images to be downloaded
    # print urls is to print the image file url
    # size is the image size which can
    # be specified manually ("large, medium, icon")
    # aspect ratio denotes the height width ratio
    # of images to download. ("tall, square, wide, panoramic")
    arguments = {"keywords": query,
                 "format": "jpg",
                 "limit": 80,
                 "print_urls": True,
                 "size": "medium",
                 "aspect_ratio": "square"}
    try:
        response.download(arguments)

        # Handling File NotFound Error
    except FileNotFoundError:
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit": 1000,
                     "print_urls": True,
                     "size": "medium"}

        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments)
        except:
            pass


# Driver Code
for query in search_queries:
    downloadimages(query)
    print()
