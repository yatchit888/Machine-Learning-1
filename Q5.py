import requests
from bs4 import BeautifulSoup
import csv

# Function to scrape and store restaurant information and reviews
def scrape_and_store_restaurant_info(url_list, output_csv):
    reviews = []

    for url in url_list:
        # Get the HTML content of the page
        response = requests.get(url)

        # If the request was successful, proceed
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get the restaurant name and total reviews
            restaurant_name = soup.find('h1', class_='css-1se8maq').text.strip()
            total_reviews = soup.find('a', class_='css-19v1rkv').text.strip()

            # Get the review text, reviewer and rating
            review_elements = soup.find_all('li', class_='css-1q2nwpv')
            for review_element in review_elements:
                review_text = review_element.find('span', {'lang': 'en'}).text.strip()
                reviewer = review_element.find('a', class_='css-19v1rkv').text.strip()
                rating = review_element.find('div', class_='five-stars__09f24__mBKym five-stars--regular__09f24__DgBNj css-1jq1ouh')['aria-label'].split()[0]
                reviews.append([restaurant_name, total_reviews, review_text, reviewer, rating])
        
        # If the request was unsuccessful, print an error message
        else:
            print(f'Failed to retrieve data from {url}')
    
    #write to csv file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Restaurant Name', 'Total Reviews', 'Review Text', 'Reviewer', 'Rating'])
        writer.writerows(reviews)
        print(f'Restaurant information and reviews have been scraped and saved to {output_csv}')
    

# Define the input and output files
output_csv_file = 'restaurant_reviews.csv'

# Define the list of URLs to scrape
restaurant_list = ['https://www.yelp.ca/biz/pai-northern-thai-kitchen-toronto-5?osq=Restaurants',
                   'https://www.yelp.ca/biz/ramen-misoya-toronto-toronto-2?osq=Restaurants',
                   'https://www.yelp.ca/biz/hana-don-toronto-2?osq=Restaurants'
                   ]

# Call the function
scrape_and_store_restaurant_info(restaurant_list, output_csv_file)