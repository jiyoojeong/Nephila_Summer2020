import googlemaps
import pprint
import time
import pandas as pd

# define markel dataset
pd.read_csv('/Users/jiyoojeong/desktop/C/Markel_Occupancy_Codes/locations_v3.csv')


# everything is returned as a dictionary!

# define api key
API_KEY = 'AIzaSyARuzgPS7RKGVSlfPGYIqpYCoajal9Xoxk'

# define client
gmaps = googlemaps.Client(key=API_KEY)

# define search

coordinates = latitude + ',' + longitude
address = #TODO:

places_result = gmaps.places_nearby(location=coordinates, radius=500, keyword=address)
pprint.pprint(places_result)

# pause script
time.sleep(2)

# get next 20 results
places_next = gmaps.places_nearby(page_token=places_result['next_page_token'])
#get pref from search
pref = places_result['photos']

# get photo from photo reference
https://maps.googleapis.com/maps/api/place/photo?photoreference=pref

#get type from search
types = places_result['type'] #--> map types to occ types and similar ones maybe to categorize

# cloud vision api?

# look at images from requests
# test out one to see if it shows empty lot or house or building

