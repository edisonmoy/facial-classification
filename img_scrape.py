# importing google_images_download module
from google_images_download import google_images_download

# creating object
response = google_images_download.googleimagesdownload()

search_queries = ['chinese man face', 'chinese woman face', 'chinese man', 'chinese woman',
                  'chinese adult face', 'chinese person', 'chinese person face', 'chinese adult', 'chinese child',
                  'ghanaian man face', 'ghanaian woman face', 'ghanaian male face', 'ghanaian woman',
                  'ghanaian adult face', 'ghanaian person face', 'ghanaian person', 'ghanaian adult', 'ghanaian child']


# search_queries = ['happy person', 'happy man', 'happy woman', 'happy child', 'happy smiling person',
#                   'happy smiling man', 'happy smiling woman', 'happy smiling child',
#                   'sad person', 'sad man', 'sad woman', 'sad child', 'sad frowning person',
#                   'sad frowning man', 'sad frowning woman', 'sad frowning child',
#                   ]


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
            response.download(arguments)
        except:
            pass


# Driver Code
for query in search_queries:
    downloadimages(query)
    print()
